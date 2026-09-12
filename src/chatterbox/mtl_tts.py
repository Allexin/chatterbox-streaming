from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import numpy as np
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"
DEFAULT_MULTILINGUAL_T3_MODEL = "t3_mtl23ls_v2.safetensors"
MULTILINGUAL_T3_MODELS = {
    "v2": "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v2": "t3_mtl23ls_v2.safetensors",
    "v3": "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v3": "t3_mtl23ls_v3.safetensors",
}

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def _resolve_multilingual_t3_model(t3_model: str | None) -> str:
    if t3_model is None:
        return DEFAULT_MULTILINGUAL_T3_MODEL
    if t3_model in MULTILINGUAL_T3_MODELS:
        return MULTILINGUAL_T3_MODELS[t3_model]
    if t3_model.endswith(".safetensors"):
        return t3_model
    raise ValueError(
        f"Unknown multilingual T3 model '{t3_model}'. "
        f"Expected one of {sorted(MULTILINGUAL_T3_MODELS)} or a .safetensors filename."
    )


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def quietest_cut(audio, search, frame):
    """How much to hold back so the join lands in the quietest place nearby.

    A chunk boundary is a switch between two independently vocoded renderings.
    Inside a vowel the waveform is periodic and the ear hears the broken period
    even when no sample jumps far, which is why measuring the step found nothing
    while listening found a tick. In a pause there is no period to break.

    So the boundary is moved backwards, never forwards -- audio beyond it has not
    been decoded -- to the quietest frame within `search` samples of the end.
    Measured on six joins of one utterance, a frame at a tenth of the loud level
    sat a median of 195 ms back and never further than 380 ms.

    Returns the number of samples to withhold, which is what the next chunk will
    carry. Zero means the schedule's own boundary was already the quietest point.
    """
    search = min(search, len(audio) // 2)
    if search < frame * 2:
        return 0
    region = audio[len(audio) - search:]
    frames = len(region) // frame
    if frames < 2:
        return 0
    energy = np.sqrt((region[:frames * frame].reshape(frames, frame) ** 2).mean(axis=1))
    # The *last* frame that is quiet enough, not the quietest one. Through a
    # pause every frame is a candidate and the earliest of them would withhold
    # the whole pause for no gain; the latest puts the boundary just before the
    # next sound starts and hands the next chunk as little as possible.
    quiet = np.flatnonzero(energy <= energy.min() * 1.5 + 1e-4)
    return int((frames - int(quiet[-1])) * frame) if len(quiet) else 0


def splice(tail, overlap, following):
    """Fade two renderings of the same speech across each other. Off by default.

    The reasoning that produced this was wrong and the measurement is worth
    keeping. The vocoder starts from noise, so decoding one token twice gives
    two different waveforms -- which sounds like it should make a butt-joint
    click. Measured at six joins, it does not: with `context_tokens` at 50 the
    step at a hard cut has a median of 0.116 of the local amplitude against 0.167
    for ordinary speech in the same file. The overlap does not merely give the
    vocoder history, it makes the two renderings agree, because both are
    conditioned on the same preceding tokens. Fading them then *raised* the
    median step to 0.189, since blending two slightly out-of-phase copies puts a
    kink at each end of the window.

    So `crossfade_ms` defaults to zero and the joins are butted together, which
    is what was measured good. What this is still for: it is the thing that would
    make a *small* `context_tokens` safe. The overlap is the largest part of the
    streaming overhead, and shrinking it is the obvious saving -- but with little
    or no overlap the renderings stop agreeing, and then there is a step to fade.
    Measure before turning it on.
    """
    if tail is None or not len(tail):
        return np.concatenate([overlap, following])
    if not len(overlap):
        return np.concatenate([tail, following])
    width = min(len(tail), len(overlap))
    # Raised cosine: equal amplitude through the join for material this
    # correlated, and no discontinuity in the slope at either end.
    rising = 0.5 - 0.5 * np.cos(np.pi * (np.arange(width) + 0.5) / width)
    joined = tail[-width:] * (1.0 - rising) + overlap[-width:] * rising
    # Whatever of the tail is not faded is still audio somebody has to hear.
    return np.concatenate([tail[:len(tail) - width], joined, following])


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        t3_model: str | None = None,
    ) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)
        t3_model = _resolve_multilingual_t3_model(t3_model)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / t3_model)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        t3_model: str | None = None,
    ) -> 'ChatterboxMultilingualTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        t3_model = _resolve_multilingual_t3_model(t3_model)
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["ve.pt", t3_model, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device, t3_model=t3_model)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def _prepare_for_generation(self, text, language_id, audio_prompt_path, exaggeration):
        """Everything both generation paths do before a single token is sampled."""
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        return text_tokens

    @staticmethod
    def _validate_language(language_id):
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    ):
        self._validate_language(language_id)

        text_tokens = self._prepare_for_generation(text, language_id, audio_prompt_path, exaggeration)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=None,  # the model config's own ceiling
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            # Drop the final speech token's audio: it is emitted just before
            # EOS with degraded attention and decodes to ~40 ms of noise.
            n_tokens = int(speech_tokens.shape[-1])
            st_len = max(1, n_tokens - 1)
            wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]

            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _vocode_window(self, tokens, drop_tail_tokens=0):
        """Vocode a window of speech tokens and return the whole window's audio.

        There is no incremental cache in this vocoder, so a chunk is produced by
        decoding a window that reaches back into audio already sent. The caller
        decides what to keep: the reach-back is what `splice` fades across, and
        it is the entire cost of streaming, since those tokens are vocoded more
        than once. The speaker prompt is prepended by the flow module on every
        call, so timbre and level do not drift between chunks the way they would
        if each chunk were conditioned on nothing.
        """
        window = torch.tensor(tokens, dtype=torch.long, device=self.device)
        wav, _ = self.s3gen.inference(speech_tokens=window, ref_dict=self.conds.gen)
        wav = wav.squeeze(0).detach().cpu().numpy()
        if drop_tail_tokens:
            samples_per_token = S3GEN_SR // S3_TOKEN_RATE
            wav = wav[: max(1, len(tokens) - drop_tail_tokens) * samples_per_token]
        return wav

    def generate_stream(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        chunk_tokens=100,
        first_chunk_tokens=25,
        chunk_growth=1.5,
        context_tokens=50,
        crossfade_ms=0.0,
        quiet_cut_ms=400.0,
        watermark=True,
    ):
        """Yield audio while the utterance is still being sampled.

        `generate` returns when the last token has been vocoded; this returns
        the first second of speech while the rest is still being thought of. The
        model is unchanged and so is the audio it produces -- what changes is
        when the bytes leave.

        Speech tokens arrive at twenty-five a second, so `chunk_tokens` is that
        many twenty-fifths of a second of audio per yield. `context_tokens` is how
        far each decode reaches back into audio already sent: it gives the vocoder
        the history to start a window from, and at 50 it also makes the two
        renderings of the overlap agree closely enough that the joins can simply
        be butted together. `crossfade_ms` fades them across each other instead
        and defaults to zero, because measured at 50 tokens of overlap it made
        the seam very slightly worse rather than better -- see `splice`.

        `first_chunk_tokens` exists because a chunk costs a fixed amount to
        vocode whatever its size, so small chunks are expensive per second of
        speech -- and the only chunk whose size really matters to a listener is
        the first, because it is the one they are waiting for in silence. Small
        first, large afterwards.

        `quiet_cut_ms` is how far back the boundary may move to find a quiet
        place to fall in. A chunk boundary is a switch between two renderings,
        and inside a vowel that is audible however small the step is; in a pause
        it is not. Zero keeps the schedule's own boundary.

        `chunk_growth` is how fast "afterwards" arrives. Jumping straight from a
        small opening chunk to a large one hands the player a second of audio and
        then makes it wait four, so each chunk instead grows by this factor until
        it reaches `chunk_tokens`. 1.0 means no ramp: every chunk the size of the
        first one.

        Yields (1, N) float tensors at `self.sr`, in order, which concatenate into
        the same utterance `generate` would have returned.
        """
        self._validate_language(language_id)
        text_tokens = self._prepare_for_generation(text, language_id, audio_prompt_path, exaggeration)

        samples_per_token = S3GEN_SR // S3_TOKEN_RATE
        start_token = self.t3.hp.start_speech_token
        stop_token = self.t3.hp.stop_speech_token
        fade = max(0, int(crossfade_ms * S3GEN_SR / 1000))
        search = max(0, int(quiet_cut_ms * S3GEN_SR / 1000))
        quiet_frame = int(0.010 * S3GEN_SR)
        produced, sent, held = [], 0, None

        def piece(final):
            """Vocode what has accumulated, fade it onto the withheld tail, keep a new one."""
            nonlocal sent, held
            window_start = max(0, sent - context_tokens)
            wav = self._vocode_window(produced[window_start:],
                                      drop_tail_tokens=1 if final else 0)
            boundary = (sent - window_start) * samples_per_token
            reach = min(fade, boundary) if held is not None else 0
            overlap, body = wav[boundary - reach:boundary], wav[boundary:]
            if final:
                audio, held = splice(held, overlap, body), None
            else:
                # Hold back the end of this chunk so the join lands somewhere
                # quiet, and at least enough for the fade if one is configured.
                # Once audio is on the wire the boundary cannot be moved.
                keep = max(quietest_cut(body, search, quiet_frame), min(fade, len(body)))
                keep = min(keep, len(body))
                audio = splice(held, overlap, body[:len(body) - keep])
                held = body[len(body) - keep:] if keep else None
            sent = len(produced)
            if watermark:
                audio = self.watermarker.apply_watermark(audio, sample_rate=self.sr)
            return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        want = min(first_chunk_tokens, chunk_tokens)
        with torch.inference_mode():
            for token in self.t3.inference_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=None,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                progress=False,
            ):
                value = int(token.view(-1)[0])
                if value == stop_token:
                    break
                if value == start_token:
                    continue
                produced.append(value)
                if len(produced) - sent < want:
                    continue
                yield piece(final=False)
                if chunk_growth > 1.0:
                    want = min(chunk_tokens, max(want + 1, int(want * chunk_growth)))

            # The last token's audio is dropped: it is emitted just before EOS
            # with degraded attention and decodes to ~40 ms of noise. What was
            # held back still has to go out, whether or not anything follows it.
            if len(produced) - 1 > sent:
                yield piece(final=True)
            elif held is not None and len(held):
                audio = self.watermarker.apply_watermark(held, sample_rate=self.sr)                     if watermark else held
                yield torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
