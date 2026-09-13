"""The T3 decode step on a preallocated KV cache, replayed as CUDA graphs.

Sampling one speech token runs the 30-layer Llama backbone once. Through the
HuggingFace model that is about two thousand small operations a token -- mask
construction, rotary embedding, a cache that grows by concatenation, one kernel
launch after another -- and the GPU spends much of each step waiting for the
CPU to hand it the next one. On the development machine that capped sampling at
32 tokens a second; on the inference host, whose Xeon is much slower per core,
the same step ran slower than real time while its GPU sat at 20-45 %.

A CUDA graph records the step's kernels once and replays them with a single
call. That needs every tensor to keep its shape and address, so the cache is
preallocated and the step is written here over the same HF modules and
weights: same projections, same rotary embedding, same SDPA, with the unfilled
cache slots masked out. Measured against the HF step on the same tokens, the
logits are identical to the bit, and so is the sampled sequence.

Attention reads every slot of the cache it is given, filled or not, and at
4391 slots that cost gave back the whole gain. So there is one graph per
capacity bucket over a shared buffer, and a step uses the smallest bucket that
holds its position. Past the largest bucket the cache is handed back to the HF
model as a DynamicCache and sampling continues there, unchanged in numbers --
a long utterance is slower, never cut.

Sampling itself (CFG mix, penalties, multinomial) stays eager in the caller, so
the random stream is consumed exactly as before.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

DEFAULT_CAPACITIES = (512, 1024, 1536, 2048)


class GraphedDecoder:
    def __init__(self, t3, capacities=DEFAULT_CAPACITIES, batch: int = 2):
        llama, cfg = t3.tfmr, t3.cfg
        weight = t3.speech_head.weight
        self.t3 = t3
        self.config = cfg
        self.layers, self.norm, self.rotary = llama.layers, llama.norm, llama.rotary_emb
        self.head = t3.speech_head
        self.batch, self.head_dim = batch, cfg.head_dim
        self.capacities = tuple(sorted(capacities))
        self.capacity = self.capacities[-1]
        shape = (batch, cfg.num_key_value_heads, self.capacity, cfg.head_dim)
        self.keys = [torch.zeros(shape, dtype=weight.dtype, device=weight.device) for _ in self.layers]
        self.values = [torch.zeros(shape, dtype=weight.dtype, device=weight.device) for _ in self.layers]
        self.x = torch.zeros(batch, 1, cfg.hidden_size, dtype=weight.dtype, device=weight.device)
        self.position = torch.zeros(1, dtype=torch.long, device=weight.device)
        self.slots = torch.arange(self.capacity, device=weight.device)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def _step(self, capacity: int) -> torch.Tensor:
        x = self.x
        cos, sin = self.rotary(x, self.position.view(1, 1))
        mask = (self.slots[:capacity] <= self.position).view(1, 1, 1, capacity)
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = x
            h = layer.input_layernorm(x)
            q = attn.q_proj(h).view(self.batch, 1, -1, self.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(self.batch, 1, -1, self.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(self.batch, 1, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.keys[index].index_copy_(2, self.position, k)
            self.values[index].index_copy_(2, self.position, v)
            out = F.scaled_dot_product_attention(
                q, self.keys[index][:, :, :capacity], self.values[index][:, :, :capacity],
                attn_mask=mask, scale=attn.scaling)
            x = residual + attn.o_proj(out.transpose(1, 2).reshape(self.batch, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.head(self.norm(x))[:, -1, :]

    @torch.inference_mode()
    def capture(self) -> None:
        """Record one graph per bucket. Takes about a third of a second each."""
        pool = torch.cuda.graph_pool_handle()
        for capacity in reversed(self.capacities):
            # Warm-up writes land in the bucket's last slot, which any real run
            # overwrites before it can read it.
            self.position.fill_(capacity - 1)
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._step(capacity)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                logits = self._step(capacity)
            self.graphs[capacity] = (graph, logits)

    def holds(self, position: int) -> bool:
        return position < self.capacity

    @torch.inference_mode()
    def load(self, past: DynamicCache, length: int) -> None:
        """Copy the prefill's cache into the static buffer."""
        for index, layer in enumerate(past.layers):
            self.keys[index][:, :, :length].copy_(layer.keys)
            self.values[index][:, :, :length].copy_(layer.values)

    def __call__(self, embed: torch.Tensor, position: int) -> torch.Tensor:
        """Logits for the token at `position`. Valid until the next call."""
        capacity = next(size for size in self.capacities if position < size)
        graph, logits = self.graphs[capacity]
        self.x.copy_(embed)
        self.position.fill_(position)
        graph.replay()
        return logits

    @torch.inference_mode()
    def to_dynamic_cache(self, length: int) -> DynamicCache:
        """The first `length` slots as a cache the HF model can continue from."""
        cache = DynamicCache(config=self.config)
        for index in range(len(self.layers)):
            cache.update(self.keys[index][:, :, :length].clone(),
                         self.values[index][:, :, :length].clone(), index)
        return cache
