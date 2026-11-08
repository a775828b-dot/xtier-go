# Dory 1.0 (NVIDIA Nemotron-Labs, looped nGPT) for the QNN static LLM flow (static_llama.py conventions).
#
# Every residual state is a unit vector: a sublayer computes its output b, then
#   h = justnorm((1 - lr) * x + lr * justnorm(b))       (nGPT's slerp with x already unit)
# justnorm(v) = v / sqrt(|v|^2 + 1e-6) is an RMSNorm with eps 1e-6 / D and weight 1 / sqrt(D), so every norm here
# is torch.nn.RMSNorm (one HTP RmsNorm op) and lr / sqrt(D) is folded into the weight of the norm of b. The
# checkpoint scales are folded at conversion (convert_weights.py): suv_gate / suv_up into the gate / up rows,
# s_gate into the gate-projection rows, sqk^2 and 1 / sqrt(head_dim) into qs, logit_scale into the lm_head rows.
#
# Layers: n_input sublayers, n_recurrent sublayers run n_loops times (unrolled, each loop its own copy of the
# weights and its own KV caches: per_loop), n_output sublayers. Each attention evaluation is one KV instance in
# execution order, so a 2-loop Dory 4B has 8 + 2 * 20 + 8 = 56 K and 56 V cache inputs; full and sliding instances
# differ in the mask (causal / sliding window), the RoPE table (rope_theta 1e7 / 1e4) and the cache (the context /
# a ring, see cache_slots).

import json
import math
import os
from typing import List, Optional, Tuple

import torch
from torch import nn


class DoryConfig:
    """The Hugging Face config.json of Dory plus the static-graph settings the wrappers fill in."""

    def __init__(self, hf: dict):
        self.vocab_size = hf["vocab_size"]
        self.dim = hf["hidden_size"]
        self.hidden_dim = hf["intermediate_size"]
        self.n_heads = hf["num_attention_heads"]
        self.n_kv_heads = hf["num_key_value_heads"]
        self.head_dim = hf["head_dim"]
        self.layer_types = list(hf["layer_types"])
        self.n_input = hf["n_input_layers"]
        self.n_recurrent = hf["n_recurrent_layers"]
        self.n_output = hf["n_output_layers"]
        self.n_loops = hf["n_recurrent_loops"]
        assert hf["recurrent_kv_cache_mode"] == "per_loop", "only per_loop KV is exported"
        assert self.n_input + self.n_recurrent + self.n_output == len(self.layer_types)
        self.sliding_window = hf["sliding_window"]
        rp = hf["rope_parameters"]
        self.rope_theta = rp["full_attention"]["rope_theta"]
        self.local_rope_theta = rp["sliding_attention"]["rope_theta"]
        self.initializer_range = hf["initializer_range"]
        self.ngpt_alpha_init = hf["ngpt_alpha_init"]
        self.gate_logit_scale_init = hf.get("attention_output_gate_logit_scale_init")
        # static graph settings
        self.max_batch_size = 1
        self.max_seq_len = 2048
        self.max_context_len = 2048
        self.use_kv_cache = True
        self.ar_len = 1
        self.kv_io_bit_width = 32
        self.enable_masked_softmax = True
        self.enable_r3 = False
        self.embedding_scale_factor = 1.0   # read by the wrappers for every Hugging Face model
        # the sliding layers' ring: the window and 64 positions of slack (a verify step of up to 64 rows, a rewind
        # of the generation prompt's last tokens)
        self.ring_slots = self.sliding_window + 64

    @classmethod
    def from_json(cls, path: str) -> "DoryConfig":
        with open(path) as f:
            return cls(json.load(f))

    def unrolled(self) -> List[int]:
        """The physical sublayer of every evaluation in execution order: the input sublayers, the recurrent ones
        once per loop, the output ones (16 + 2 x 40 + 16 = 112 for Dory 4B)."""
        n_in, n_rec = self.n_input, self.n_recurrent
        return (list(range(n_in)) + list(range(n_in, n_in + n_rec)) * self.n_loops +
                list(range(n_in + n_rec, len(self.layer_types))))

    def loop_starts(self) -> List[int]:
        """Evaluation indices where a loop begins (the first after the input sublayers, then one per loop)."""
        return [self.n_input + k * self.n_recurrent for k in range(self.n_loops)]


def unit_norm(dim: int) -> nn.RMSNorm:
    """justnorm over the last dim (the weight is set by the checkpoint: 1 / sqrt(dim), or lr / sqrt(dim))."""
    return nn.RMSNorm(dim, eps=1e-6 / dim)


def precompute_freqs(head_dim: int, max_context_len: int, theta: float):
    """cos / sin [max_context_len, head_dim / 2] for the half-split (NeoX / Hugging Face) RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(max_context_len, dtype=torch.float32), inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, freqs_cos, freqs_sin):
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, None, :, :]
        freqs_sin = freqs_sin[None, None, :, :]
        if ATTENTION == "split":
            # the tables at x's full shape. Past the first shard they are graph inputs, and the HTP compiles their
            # broadcast over the heads wrong ("elementwise op applied to mismatched shapes" at compile time; Dory
            # tiny on the Y900: argmax agrees with the cat graph 31% of the time, 96.5% with the repeat). The cat
            # graph gets the backend's MHA -> SHA split: one head per op, nothing to broadcast.
            freqs_cos = freqs_cos.repeat(1, x.shape[1], 1, 1)
            freqs_sin = freqs_sin.repeat(1, x.shape[1], 1, 1)
    return torch.cat([x_r * freqs_cos - x_i * freqs_sin, x_r * freqs_sin + x_i * freqs_cos], dim=-1)


# the attention's graph (XTG_DORY_ATTN, read when the model is built): "cat" (the cache and the new rows concatenated,
# K / V repeated per q head) or "split" (grouped q heads, the caches read in place). 1-loop Dory on the Y900, a decode
# step: split 64 / 80 / 680 ms at 4K / 8K / 32K, cat 72 / - / 280 ms (the HTP schedules split's big grouped products
# badly at 32K); prefill split 342 / 162 / 35 tok/s, cat 478 / - / 54. The app's buckets: split 4K and 8K, cat 32K
# (one calibration's KV encodings, so a session moves between them)
ATTENTION = os.environ.get("XTG_DORY_ATTN", "cat")


def repeat_kv(h: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return h
    b, n_kv, a, c = h.shape
    return h[:, :, None, :, :].expand(b, n_kv, n_rep, a, c).reshape(b, n_kv * n_rep, a, c)


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, bias=False)


def to_conv(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C] -> [B, C, 1, T], the layout of a 1x1 conv as a linear layer"""
    b, t, c = x.shape
    return x.reshape(b, t, 1, c).transpose(1, 3)


def from_conv(y: torch.Tensor) -> torch.Tensor:
    """[B, C, 1, T] -> [B, T, C]"""
    return y.permute(0, 3, 1, 2).squeeze(-1)


class Residual(nn.Module):
    """h = justnorm((1 - lr) * x + lr * justnorm(b)): keep = 1 - lr, b_norm.weight = lr / sqrt(D). x is unit except
    for the first sublayer, whose input is an embedding row (norms 0.9996 .. 1.0013): x_norm there."""

    def __init__(self, dim: int, norm_x: bool = False):
        super().__init__()
        self.keep = nn.Parameter(torch.ones(dim))
        self.b_norm = unit_norm(dim)
        self.out_norm = unit_norm(dim)
        self.x_norm = unit_norm(dim) if norm_x else None

    def forward(self, x, b):
        if self.x_norm is not None:
            x = self.x_norm(x)
        return self.out_norm(x * self.keep + self.b_norm(b))


class DoryAttention(nn.Module):
    attention = None   # the wrappers look for prepare_*_conv on these; Dory is written with convs already
    feed_forward = None

    def __init__(self, cfg: DoryConfig, sliding: bool, first: bool = False):
        super().__init__()
        self.sliding = sliding
        self.dim, self.n_heads, self.n_kv_heads, self.head_dim = cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.enable_masked_softmax = cfg.enable_masked_softmax
        qd, kd = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        self.wq, self.wk, self.wv = conv1x1(cfg.dim, qd), conv1x1(cfg.dim, kd), conv1x1(cfg.dim, kd)
        self.wg = conv1x1(cfg.dim, qd)   # s_gate^2 folded into the rows
        self.wo = conv1x1(qd, cfg.dim)
        self.q_norm, self.k_norm = unit_norm(cfg.head_dim), unit_norm(cfg.head_dim)
        # sqk^2 / sqrt(head_dim) per q head, [1, heads, 1, head_dim]: the head on axis 1 like q, so the backend's
        # MHA -> SHA pass splits it with q (a [heads, 1, head_dim] parameter is not split and breaks the pass)
        self.qs = nn.Parameter(torch.ones(1, cfg.n_heads, 1, cfg.head_dim))
        self.o_norm = unit_norm(qd)
        self.res = Residual(cfg.dim, norm_x=first)
        self.enable_r3 = cfg.enable_r3
        if self.enable_r3:
            import scipy.linalg

            self.register_buffer(
                "r3_weight",
                torch.tensor(scipy.linalg.hadamard(cfg.head_dim, dtype=float) / math.sqrt(cfg.head_dim),
                             dtype=torch.float32),
                persistent=False,
            )

    def forward(self, x, freqs_cos, freqs_sin, atten_mask, k_cache=None, v_cache=None):
        bsz, seq_len, _ = x.shape
        xc = to_conv(x)
        q = from_conv(self.wq(xc)).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = from_conv(self.wk(xc)).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = from_conv(self.wv(xc)).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        gate = from_conv(self.wg(xc))
        # justnorm commutes with RoPE (a rotation), so the norm goes first and RoPE sees unit vectors
        q = apply_rope(self.q_norm(q), freqs_cos, freqs_sin) * self.qs
        k = apply_rope(self.k_norm(k), freqs_cos, freqs_sin)
        if self.enable_r3:
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)
        if ATTENTION == "cat":
            # the cache and the new rows concatenated, K / V repeated per q head: what runs on the Y900 today; at 32K
            # the HTP copies the caches every step (the concat and the repeat)
            kh = torch.cat([k_cache, k], dim=-1)   # this order (both concats, then both repeats) is the graph the
            vh = torch.cat([v_cache, v], dim=2)    # saved calibrations were made on: the observers go by position
            kh = repeat_kv(kh, self.n_rep)
            vh = repeat_kv(vh, self.n_rep)
            attn = q @ kh   # qs carries 1 / sqrt(head_dim)
            if self.enable_masked_softmax:
                attn_min = torch.amin(attn, dim=-1, keepdim=True)
                attn = torch.where(atten_mask == 0, attn, attn_min - 20)
            else:
                attn = attn + atten_mask
            attn = torch.softmax(attn, dim=-1)
            y = (attn @ vh).transpose(1, 2).reshape(bsz, seq_len, -1)
            y = self.o_norm(y * torch.sigmoid(gate))
            out = from_conv(self.wo(to_conv(y)))
            return self.res(x, out), k, v
        # grouped-query attention without a copy of K / V per q head (expanding them materialized n_rep copies on the
        # HTP, about 1 GB per layer at 32K): the n_rep q heads of a KV head (heads kv * n_rep + r) go along the rows,
        # the mask repeated with them
        q = q.reshape(bsz, self.n_kv_heads, self.n_rep * seq_len, self.head_dim)
        # the mask at the scores' full shape, KV heads too: this graph gets no MHA -> SHA split, so nothing should
        # broadcast over the heads (see apply_rope)
        mask = atten_mask.repeat(1, self.n_kv_heads, self.n_rep, 1)
        # the cache and the new rows apart (scores joined, probabilities split): a cat of the cache with the new K / V
        # copies the whole cache every step on the HTP. qs carries 1 / sqrt(head_dim). The calibration graph's caches
        # have no slots, which keeps the same nodes (DoryQuantRecipe shares the cache matmuls' encodings).
        slots = k_cache.shape[-1] if k_cache is not None else 0
        if k_cache is None:
            k_cache = k.new_zeros(bsz, self.n_kv_heads, self.head_dim, 0)
            v_cache = v.new_zeros(bsz, self.n_kv_heads, 0, self.head_dim)
        attn = torch.cat([q @ k_cache, q @ k], dim=-1)
        if self.enable_masked_softmax:
            # Dory's scores reach the hundreds: an additive -255 mask would let masked keys win
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            attn = torch.where(mask == 0, attn, attn_min - 20)
        else:
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        # split, not sliced: with no cache slots (the calibration graph) a slice of everything exports as an alias,
        # and every graph must have the same nodes. The new rows' product first: the cache product shares its
        # encoding, and a shared encoding may only refer to a node earlier in the graph.
        p_cache, p_new = torch.split(attn, [slots, attn.shape[-1] - slots], dim=-1)
        y = p_new @ v + p_cache @ v_cache
        y = y.reshape(bsz, self.n_heads, seq_len, self.head_dim)
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        y = self.o_norm(y * torch.sigmoid(gate))
        out = from_conv(self.wo(to_conv(y)))
        return self.res(x, out), k, v


class DoryMLP(nn.Module):
    attention = None
    feed_forward = None

    def __init__(self, cfg: DoryConfig, first: bool = False):
        super().__init__()
        self.wg = conv1x1(cfg.dim, cfg.hidden_dim)   # suv_gate folded
        self.wu = conv1x1(cfg.dim, cfg.hidden_dim)   # suv_up folded
        self.wd = conv1x1(cfg.hidden_dim, cfg.dim)
        self.m_norm = unit_norm(cfg.hidden_dim)
        self.res = Residual(cfg.dim, norm_x=first)

    def forward(self, x):
        xc = to_conv(x)
        m = from_conv(nn.functional.silu(self.wg(xc)) * self.wu(xc))
        out = from_conv(self.wd(to_conv(self.m_norm(m))))
        return self.res(x, out)


class DoryModel(nn.Module):
    """forward(tokens, atten_mask, window_atten_mask, input_pos, *k_caches, *v_caches)
    -> (logits, new_k_caches, new_v_caches), caches in execution order (self.kv)."""

    def __init__(self, config: DoryConfig, ar_len=1, output_new_cache_only=True, output_cache=True,
                 use_i64_token=False, **kwargs):
        super().__init__()
        assert output_new_cache_only, "Dory exports the new cache slices only"
        self.config = config
        self.dim, self.head_dim, self.vocab_size = config.dim, config.head_dim, config.vocab_size
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.max_batch_size, self.max_seq_len = config.max_batch_size, config.max_seq_len
        self.max_context_len, self.use_kv_cache = config.max_context_len, config.use_kv_cache
        self.ar_len, self.output_cache, self.use_i64_token = ar_len, output_cache, use_i64_token
        self.kv_io_bit_width = config.kv_io_bit_width
        self.sliding_window = config.sliding_window
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        # one module per evaluation: the loops are unrolled with their own copy of the recurrent weights. The HTP
        # compiler turns a weight that two ops read into a DDR tensor recomputed every step (1 byte per parameter
        # of the looped block, so the context does not fit and the step reads it twice); separate copies cost the
        # memory once and nothing per step. Execution order = index order, so the backend's sharding by index
        # cuts between loops.
        self.phys = config.unrolled()
        types = config.layer_types
        self.sublayers = nn.ModuleList(
            [DoryMLP(config, i == 0) if types[p] == "mlp" else DoryAttention(config, types[p] == "sliding_attention", i == 0)
             for i, p in enumerate(self.phys)]
        )
        self.loop_starts = config.loop_starts()
        self.loop_norm = unit_norm(config.dim)   # justnorm(state + input) at the start of loops 2..
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)   # logit_scale folded
        self.kv = [i for i, p in enumerate(self.phys) if types[p] != "mlp"]
        self.n_layers = len(self.kv)   # the runner's "layers" are the KV instances
        cos, sin = precompute_freqs(config.head_dim, config.max_context_len, config.rope_theta)
        lcos, lsin = precompute_freqs(config.head_dim, config.max_context_len, config.local_rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)
        self.register_buffer("local_freqs_cos", lcos, persistent=False)
        self.register_buffer("local_freqs_sin", lsin, persistent=False)

    @property
    def layers(self):
        return list(self.sublayers)

    def forward(self, tokens, atten_mask, window_atten_mask, input_pos: Optional[torch.Tensor] = None, *args):
        cfg = self.config
        if self.use_kv_cache:
            cos, sin = self.freqs_cos[input_pos][0], self.freqs_sin[input_pos][0]
            lcos, lsin = self.local_freqs_cos[input_pos][0], self.local_freqs_sin[input_pos][0]
        else:
            cos, sin, lcos, lsin = self.freqs_cos, self.freqs_sin, self.local_freqs_cos, self.local_freqs_sin
        n = self.n_layers
        k_caches = list(args[:n]) if self.use_kv_cache else [None] * n
        v_caches = list(args[n: 2 * n]) if self.use_kv_cache else [None] * n
        out_k, out_v = [], []

        def run(i, h):
            layer = self.sublayers[i]
            if isinstance(layer, DoryMLP):
                return layer(h)
            s = len(out_k)
            if layer.sliding:
                h, k, v = layer(h, lcos, lsin, window_atten_mask, k_caches[s], v_caches[s])
            else:
                h, k, v = layer(h, cos, sin, atten_mask, k_caches[s], v_caches[s])
            out_k.append(k)
            out_v.append(v)
            return h

        h = self.tok_embeddings(tokens)
        input_h = None
        for i in range(len(self.sublayers)):
            if i == self.loop_starts[0]:
                input_h = h
            elif i in self.loop_starts:
                # loop 0 starts from justnorm(2 input) = input; the state after a loop is already unit
                h = self.loop_norm(h + input_h)
            h = run(i, h)
        logits = self.output(h)
        if self.output_cache:
            return logits, out_k, out_v
        return logits

    def cache_slots(self):
        """(full, ring) KV slots of one instance. The prefill and decode graphs share one layout: a full-attention
        instance keeps every position of the context (slot = position), a sliding one a ring of ring_slots (slot =
        position % ring_slots, window + slack so a short rewind finds its window intact); the new rows are
        appended in the graph and written back by the runner (xtg_npu). The calibration graph (AR = context)
        starts from empty caches."""
        if self.ar_len == self.max_context_len:
            return 0, 0
        return self.max_context_len, self.config.ring_slots

    def get_example_inputs(self):
        # here, not at the top: the llama package imports this module when it registers Dory
        from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
            AttentionMask,
            BaseAttentionMask,
            CausalAttentionMask,
            SlidingWindowAttentionMask,
        )

        class StaticMask(BaseAttentionMask):
            """A mask of the shared layout: [batch, 1, AR, cache slots + AR], filled by the runner every step."""

            def __init__(self, b, ar, cols):
                super().__init__(b, ar, cols)
                self._mask = torch.zeros(b, 1, ar, cols)

            @property
            def mask(self):
                return self._mask

            def smart_mask_init(self, pos):
                pass

            def smart_mask_update(self, pos, n_updates, lade_pos_offset):
                pass

        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype)
        full, ring = self.cache_slots()
        b, ar = self.max_batch_size, self.ar_len
        if full == 0:
            mask = AttentionMask([CausalAttentionMask(b, ar, self.max_context_len),
                                  SlidingWindowAttentionMask(b, ar, self.max_context_len,
                                                             sliding_window=self.sliding_window)])
        else:
            mask = AttentionMask([StaticMask(b, ar, full + ar), StaticMask(b, ar, ring + ar)])
        if not self.use_kv_cache:
            return (tokens, mask)
        pos_ids = torch.zeros((b, ar), dtype=torch.int32)
        slots = [ring if self.sublayers[i].sliding else full for i in self.kv]
        k_caches = [torch.zeros(b, self.n_kv_heads, self.head_dim, s) for s in slots]
        v_caches = [torch.zeros(b, self.n_kv_heads, s, self.head_dim) for s in slots]
        return (tokens, mask, pos_ids, k_caches, v_caches)

    def get_metadata(self):
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 11,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_max_context_len": self.max_context_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_kv_io_bit_width": self.kv_io_bit_width,
            "get_sliding_window": self.sliding_window,
            "get_ring_slots": self.config.ring_slots,
        }
