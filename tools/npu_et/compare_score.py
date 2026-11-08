"""compare_score.py TOKENS.bin SCORE.txt: the NPU's next-token NLL and argmax (npu_test ... score) against
reference/dory_ref.py over the same tokens (BF16 weights in FP32): mean NLL of both, argmax agreement, by part.

  DORY_BF16=... python compare_score.py tokens.bin score.txt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
import dory_ref  # noqa: E402

ids = np.fromfile(sys.argv[1], dtype=np.uint64).astype(np.int64).tolist()
rows = [l.split() for l in open(sys.argv[2])]
npu_arg = torch.tensor([int(r[0]) for r in rows])
npu_nll = torch.tensor([float(r[1]) for r in rows])
n = len(rows)
torch.set_num_threads(64)
dory_ref.dequant = lambda f, name, device: f.get_tensor(name + ".weight").float().to(device)
lg = dory_ref.Ref(os.environ["DORY_BF16"]).forward(ids[:n])
lp = torch.log_softmax(lg, -1)
nxt = torch.tensor(ids[1:n])
ref_nll = -lp[:-1].gather(1, nxt[:, None]).squeeze(1)
agree = (lg.argmax(-1) == npu_arg).float()
print(f"positions {n}: next-token NLL ref {ref_nll.mean():.4f} npu {npu_nll[:-1].mean():.4f} "
      f"(PPL {ref_nll.mean().exp():.3f} -> {npu_nll[:-1].mean().exp():.3f}, "
      f"{((npu_nll[:-1].mean() - ref_nll.mean()).exp().item() - 1) * 100:+.1f}%), argmax agree {agree.mean():.4f}")
for a, b in ((0, n // 4), (n // 4, n // 2), (n // 2, n)):
    print(f"  {a}..{b}: agree {agree[a:b].mean():.4f}  NLL ref {ref_nll[a:min(b, n - 1)].mean():.4f} "
          f"npu {npu_nll[a:min(b, n - 1)].mean():.4f}")
