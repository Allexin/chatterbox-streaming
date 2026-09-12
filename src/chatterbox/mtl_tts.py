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


def audio_lead(prompt_feat_frames, prompt_tokens, token_mel_ratio, per_token):
    """Samples by which a window's audio starts after the token it was asked to start at.

    `flow.inference` returns the frames after `mel_len1`, the prompt's mel
    length, while the generated frames begin at `prompt_tokens x token_mel_ratio`.
    For the reference voice those are 411 and 410, so every window's audio starts
    one frame -- half a token, 20 ms -- late. Streaming that ignored it skipped
    20 ms of speech at every join: measured 2026-09-12, the stream ran 0, 20, 40
    and 60 ms ahead of a whole decode of the same tokens across its first four
    pieces. Inside a word that is a stutter. The lead depends on the voice, and a
    prompt whose mel is shorter than its tokens gives a negative one.
    """
    per_frame = per_token // token_mel_ratio
    return (prompt_feat_frames - prompt_tokens * token_mel_ratio) * per_frame


def emit_from(sent, window_start, per_token, lead):
    """Where, in a window's own audio, the first token not yet sent begins."""
    return max(0, (sent - window_start) * per_token - lead)


def quietest_boundary(audio, search, per_token, lookahead=3):
    """How many whole tokens to leave to the next window, so the switch lands in a pause.

    A piece boundary is a switch between two independently rendered windows,
    and inside a word that switch is audible; in a pause it is not. So the piece
    ends on a token boundary -- the one place two renderings of one token
    sequence line up by construction -- and the next window renders from there.
    Holding audio back instead, as an earlier version did, only moves where the
    bytes are cut: withheld audio is still the same window's.

    Each candidate boundary is scored by the level of the two tokens around it,
    80 ms, which is long enough that the closure of a stop consonant inside a
    word does not pass for a pause. The latest boundary within 1.5x of the
    quietest wins: through a pause every boundary qualifies, and the latest
    hands the next window the least.

    Never fewer than `lookahead` tokens are left: the flow encoder looks three
    tokens ahead, and a window's last three are rendered without that, so they
    are the worst audio in it and are better re-rendered by the next window.
    Never more than half of what is on offer.
    """
    tokens = len(audio) // per_token
    half = tokens // 2
    most = min(search // per_token, half)
    if most < lookahead:
        return min(lookahead, half)
    levels = []
    for kept_back in range(lookahead, most + 1):
        boundary = (tokens - kept_back) * per_token
        around = audio[boundary - per_token:boundary + per_token]
        levels.append(float(np.sqrt(np.mean(around ** 2))))
    levels = np.asarray(levels)
    # Without a real pause in reach, "the latest boundary near the quietest" is
    # just the latest boundary: when everything is loud, everything is within
    # 1.5x of the minimum. Measured at 100-token pieces, that put two switches
    # at 0.98 and 0.66 of the loud level. Then the quietest one it is.
    width = 2 * per_token
    frames = audio[:len(audio) // width * width].reshape(-1, width)
    loud = float(np.percentile(np.sqrt((frames ** 2).mean(axis=1)), 90)) if len(frames) else 0.0
    if levels.min() > 0.1 * loud:
        return lookahead + int(np.argmin(levels))
    quiet = np.flatnonzero(levels <= levels.min() * 1.5 + 1e-4)
    return lookahead + int(quiet[0])


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

    def _vocode_window(self, tokens, drop_tail_tokens=0, lead_samples=0):
        """Vocode a window of speech tokens and return its audio.

        A piece is produced by decoding a window that reaches back into audio
        already sent, and the caller decides what to keep. The reach-back gives
        the flow decoder history to start from, and it is the cost of streaming,
        since those tokens are vocoded more than once. The speaker prompt is
        prepended by the flow module on every call, so timbre and level do not
        drift between pieces. Two renderings of the same tokens are never the
        same waveform, which is why the caller drops the overlap rather than
        joining to it.
        """
        window = torch.tensor(tokens, dtype=torch.long, device=self.device)
        wav, _ = self.s3gen.inference(speech_tokens=window, ref_dict=self.conds.gen)
        wav = wav.squeeze(0).detach().cpu().numpy()
        if drop_tail_tokens:
            samples_per_token = S3GEN_SR // S3_TOKEN_RATE
            # Token boundaries sit `lead_samples` earlier in this audio than
            # their count says; see `audio_lead`.
            wav = wav[: max(1, (len(tokens) - drop_tail_tokens) * samples_per_token - lead_samples)]
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
        pause_search_ms=0.0,
        max_new_tokens=None,
        watermark=True,
    ):
        """Yield audio while the utterance is still being sampled.

        `generate` returns when the last token has been vocoded; this returns
        speech while the rest is still being thought of. The model is unchanged;
        what changes is when the bytes leave.

        Speech tokens arrive at twenty-five a second, so `chunk_tokens` is that
        many twenty-fifths of a second of audio per yield. `context_tokens` is how
        far each decode reaches back into audio already sent, to give the flow
        decoder history to start from; the overlap is rendered again and dropped.

        `first_chunk_tokens` and `chunk_growth` let the opening piece be smaller
        than the rest and grow towards `chunk_tokens`. The service ships them
        equal, at 100 (owner decision 2026-09-12): the opening piece is the
        player's whole margin, and a pause is only within reach of a large piece.

        `pause_search_ms` is how far back a piece may end so that the switch
        between two renderings falls in a pause; see `quietest_boundary`. Zero
        ends every piece on the schedule, which puts the switch in sound five
        times in seven. With it, the owner could not hear the joins.

        Yields (1, N) float tensors at `self.sr`, in order.
        """
        self._validate_language(language_id)
        text_tokens = self._prepare_for_generation(text, language_id, audio_prompt_path, exaggeration)

        samples_per_token = S3GEN_SR // S3_TOKEN_RATE
        start_token = self.t3.hp.start_speech_token
        stop_token = self.t3.hp.stop_speech_token
        pause_search = max(0, int(pause_search_ms * S3GEN_SR / 1000))
        # Every window's audio starts this many samples after its first token,
        # and every boundary below is counted in that window's own samples.
        lead = audio_lead(self.conds.gen["prompt_feat"].size(1),
                          self.conds.gen["prompt_token"].size(-1),
                          self.s3gen.flow.token_mel_ratio, samples_per_token)
        produced, sent = [], 0

        def piece(final):
            """Vocode what has accumulated and emit what has not been sent."""
            nonlocal sent
            window_start = max(0, sent - context_tokens)
            wav = self._vocode_window(produced[window_start:],
                                      drop_tail_tokens=1 if final else 0, lead_samples=lead)
            audio = wav[emit_from(sent, window_start, samples_per_token, lead):]
            emitted_to = len(produced)
            if pause_search and not final:
                # End on a quiet token boundary and let the next window render
                # from there, so the switch itself falls in the pause.
                kept_back = quietest_boundary(audio, pause_search, samples_per_token)
                audio = audio[:len(audio) - kept_back * samples_per_token]
                emitted_to -= kept_back
            sent = emitted_to
            if watermark:
                audio = self.watermarker.apply_watermark(audio, sample_rate=self.sr)
            return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        want = min(first_chunk_tokens, chunk_tokens)
        with torch.inference_mode():
            for token in self.t3.inference_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
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
            # with degraded attention and decodes to ~40 ms of noise.
            if len(produced) - 1 > sent:
                yield piece(final=True)
