"""quant_sim.py TOKENS.bin: which part of the NPU export's quantization costs Dory its accuracy. The reference model
(reference/dory_ref.py, FP32) with fake quantization of one part at a time, mean next-token NLL over the tokens:
  kv8 / k8 / v8   K and / or V at 8 bits, one static range per KV instance (the range over these tokens, as a
                  calibration would find it)
  kv16            both at 16 bits
  w4              linear weights 4-bit integers in blocks of 16 along the input, symmetric (16a4w_block)
  w4kv8 / w4k16v8 the export's mix / a cheaper fix

  DORY_BF16=... python quant_sim.py prompts/dory-tail2048.bin [N 2048] [cases ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
import dory_ref  # noqa: E402

D = os.environ["DORY_BF16"]
ids = np.fromfile(sys.argv[1], dtype=np.uint64).astype(np.int64).tolist()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
cases = sys.argv[3:] or ["base", "k8", "v8", "kv8", "kv16", "w4", "w4kv8", "w4k16v8"]
ids = ids[:N]
dev = "cuda" if torch.cuda.is_available() else "cpu"
dory_ref.dequant = lambda f, name, device: f.get_tensor(name + ".weight").float().to(device)


def fq(x, lo, hi, bits):
    """asymmetric fake quantization to `bits` over [lo, hi]"""
    q = (1 << bits) - 1
    s = max(hi - lo, 1e-12) / q
    return (torch.round((x - lo) / s).clamp(0, q) * s + lo).to(x.dtype)


def w4(w):
    """symmetric 4-bit, blocks of 16 along the input (last) dim"""
    o, i = w.shape
    b = w.reshape(o, i // 16, 16)
    s = b.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 7
    return (torch.round(b / s).clamp(-8, 7) * s).reshape(o, i)


ref = dory_ref.Ref(D, device=dev)
orig = {li: dict(L) for li, L in enumerate(ref.layers)}
W = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ranges = {}
mode = {"k": 0, "v": 0}
base_attention = dory_ref.Ref.attention


def attention(self, li, inst, q, k, v, pos0, store):
    key = (li, inst)
    if mode["collect"]:
        r = ranges.setdefault(key, [1e9, -1e9, 1e9, -1e9])
        r[0], r[1] = min(r[0], k.min().item()), max(r[1], k.max().item())
        r[2], r[3] = min(r[2], v.min().item()), max(r[3], v.max().item())
    else:
        r = ranges[key]
        if mode["k"]:
            k = fq(k, r[0], r[1], mode["k"])
        if mode["v"]:
            v = fq(v, r[2], r[3], mode["v"])
    return base_attention(self, li, inst, q, k, v, pos0, store)


dory_ref.Ref.attention = attention


def nll():
    ref.reset()
    lg = ref.forward(ids)
    lp = torch.log_softmax(lg.float(), -1)
    nxt = torch.tensor(ids[1:], device=lp.device)
    return -lp[:-1].gather(1, nxt[:, None]).mean().item(), lg.argmax(-1).cpu()


mode["collect"] = True
base, base_arg = nll()
mode["collect"] = False
print(f"base NLL {base:.4f}", flush=True)
for c in cases:
    if c == "base":
        continue
    quant_w = c.startswith("w4")
    for li, L in enumerate(ref.layers):
        for n in W:
            if n in orig[li]:
                L[n] = w4(orig[li][n]) if quant_w else orig[li][n]
    kv = c[2:] if quant_w else c
    mode["k"] = {"kv8": 8, "k8": 8, "kv16": 16, "k16v8": 16}.get(kv, 0)
    mode["v"] = {"kv8": 8, "v8": 8, "kv16": 16, "k16v8": 8}.get(kv, 0)
    v, arg = nll()
    print(f"{c:8s} NLL {v:.4f} ({(np.exp(v - base) - 1) * 100:+.1f}% PPL), argmax agree {(arg == base_arg).float().mean():.4f}",
          flush=True)
