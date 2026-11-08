"""test_dory_ref.py: DoryModel (the static graph, FP32) against reference/dory_ref.py (BF16 weights in FP32).

The static model runs the way libxtg_npu drives it: chunks of AR tokens; full-attention caches of CTX slots
(slot = position), sliding ones a ring of RING slots (slot = position % RING); both masks built per chunk as the
plugin builds them; the new K / V written back. The window is set to WINDOW and the ring to RING on both sides so
a short test wraps the ring several times.

  DORY_BF16=... DORY_TRACES=... python test_dory_ref.py [TOKENS 96] [AR 32] [CTX 128] [WINDOW 16] [RING 24]
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
import dory_ref  # noqa: E402
from executorch.examples.qualcomm.oss_scripts.dory.convert_weights import load_state_dict  # noqa: E402
from executorch.examples.qualcomm.oss_scripts.dory.model import DoryConfig, DoryModel  # noqa: E402

D = os.environ["DORY_BF16"]   # the BF16 checkpoint directory
T, AR, CTX, WINDOW, RING = (int(x) for x in (sys.argv[1:6] + ["96", "32", "128", "16", "24"][len(sys.argv) - 1:]))
SEE, HIDE = 0.0, -255.0
torch.set_num_threads(64)

rec = json.loads(open(os.environ["DORY_TRACES"] + "/s1-think.jsonl").readline())
ids = rec["prompt"][:T]

dory_ref.dequant = lambda f, name, device: f.get_tensor(name + ".weight").float().to(device)
t0 = time.time()
ref = dory_ref.Ref(D)
ref.window = WINDOW
ref_logits = ref.forward(ids)
print(f"reference {time.time() - t0:.1f} s")

cfg = DoryConfig.from_json(D + "/config.json")
cfg.max_context_len, cfg.max_seq_len, cfg.ar_len, cfg.sliding_window, cfg.ring_slots = CTX, CTX, AR, WINDOW, RING
model = DoryModel(cfg, ar_len=AR).eval()
model.load_state_dict(load_state_dict(D), strict=True)
n, kvh, hd = model.n_layers, cfg.n_kv_heads, cfg.head_dim
ring = [model.sublayers[i].sliding for i in model.kv]
slots = [RING if r else CTX for r in ring]
kc = [torch.zeros(1, kvh, hd, s) for s in slots]
vc = [torch.zeros(1, kvh, s, hd) for s in slots]
ring_pos = [-1] * RING
out = []
t0 = time.time()
with torch.no_grad():
    for pos0 in range(0, T, AR):
        m = min(AR, T - pos0)
        toks = torch.zeros(1, AR, dtype=torch.int32)
        toks[0, :m] = torch.tensor(ids[pos0: pos0 + m])
        full = torch.full((1, 1, AR, CTX + AR), HIDE)
        win = torch.full((1, 1, AR, RING + AR), HIDE)
        for i in range(AR):
            if i >= m:
                full[0, 0, i, CTX + i] = SEE
                win[0, 0, i, RING + i] = SEE
                continue
            p = pos0 + i
            full[0, 0, i, :pos0] = SEE
            full[0, 0, i, CTX: CTX + i + 1] = SEE
            for s in range(RING):
                q = ring_pos[s]
                if 0 <= q < pos0 and p - q < WINDOW:
                    win[0, 0, i, s] = SEE
            for j in range(max(0, i - WINDOW + 1), i + 1):
                win[0, 0, i, RING + j] = SEE
        pos = torch.arange(pos0, pos0 + AR, dtype=torch.int32)[None].clamp(max=CTX - 1)
        logits, nk, nv = model(toks, full, win, pos, *kc, *vc)
        out.append(logits[0, :m])
        for s in range(n):
            for r in range(m):
                slot = (pos0 + r) % RING if ring[s] else pos0 + r
                kc[s][..., slot] = nk[s][..., r]
                vc[s][:, :, slot] = nv[s][:, :, r]
        for r in range(m):
            ring_pos[(pos0 + r) % RING] = pos0 + r
print(f"static model {time.time() - t0:.1f} s")
got = torch.cat(out)
d = (got - ref_logits).abs()
agree = (got.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
lp_ref = torch.log_softmax(ref_logits, -1)
lp_got = torch.log_softmax(got, -1)
kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
print(f"tokens {T} ar {AR} ctx {CTX} window {WINDOW} ring {RING}: max |dlogit| {d.max().item():.2e}, "
      f"mean {d.mean().item():.2e}, argmax agree {agree:.3f}, KL max {kl.max().item():.2e}")
