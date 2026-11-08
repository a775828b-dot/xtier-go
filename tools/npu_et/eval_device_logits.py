"""eval_device_logits.py PTE TOKENS.bin DUMP.raw [N]: the NPU's prefill logits against reference/dory_ref.py.

DUMP is what qnn_llama_runner --dump_logits_path writes: the uint16 logits of every prefill chunk (AR rows each,
the last chunk padded), then the decode steps. The first N prompt positions (default all) are dequantized with the
.pte's get_logits_scale / get_logits_zero_point and compared with the reference forward over the same tokens (BF16
weights in FP32): next-token NLL of both, argmax agreement, KL(ref || npu), and the reference's top-1 among the NPU's
top-5.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
import dory_ref  # noqa: E402
from executorch.runtime import Runtime  # noqa: E402

D = os.environ["DORY_BF16"]   # the BF16 checkpoint directory
pte, tok_path, dump_path = sys.argv[1:4]
ids = np.fromfile(tok_path, dtype=np.uint64).astype(np.int64).tolist()
n = int(sys.argv[4]) if len(sys.argv) > 4 else len(ids)
ids = ids[:n]
prog = Runtime.get().load_program(pte)
scale = prog.load_method("get_logits_scale").execute([])[0]
zp = prog.load_method("get_logits_zero_point").execute([])[0]
vocab = prog.load_method("get_vocab_size").execute([])[0]
raw = np.fromfile(dump_path, dtype=np.uint16).reshape(-1, vocab)[:n]
npu = (torch.from_numpy(raw.astype(np.float32)) - zp) * scale
torch.set_num_threads(64)
dory_ref.dequant = lambda f, name, device: f.get_tensor(name + ".weight").float().to(device)
ref = dory_ref.Ref(D).forward(ids)
lr, ln = torch.log_softmax(ref, -1), torch.log_softmax(npu, -1)
nxt = torch.tensor(ids[1:])
nll_r = -lr[:-1].gather(1, nxt[:, None]).squeeze(1)
nll_n = -ln[:-1].gather(1, nxt[:, None]).squeeze(1)
agree = (ref.argmax(-1) == npu.argmax(-1)).float()
kl = (lr.exp() * (lr - ln)).sum(-1)
top5 = (npu.topk(5, -1).indices == ref.argmax(-1, keepdim=True)).any(-1).float()
print(f"positions {n}  logits scale {scale:.5f} zp {zp}  (step {scale:.4f}, range {(0 - zp) * scale:.1f}..{(65535 - zp) * scale:.1f})")
print(f"next-token NLL  ref {nll_r.mean():.4f}  npu {nll_n.mean():.4f}  (PPL {nll_r.mean().exp():.3f} -> {nll_n.mean().exp():.3f}, "
      f"{(nll_n.mean() - nll_r.mean()).exp().item() * 100 - 100:+.1f}%)")
print(f"argmax agree {agree.mean():.4f}   ref top-1 in npu top-5 {top5.mean():.4f}   KL mean {kl.mean():.4f}  p95 {kl.quantile(0.95):.4f}  max {kl.max():.3f}")
for a, b in ((0, n // 4), (n // 4, n // 2), (n // 2, n)):
    print(f"  positions {a}..{b}: agree {agree[a:b].mean():.4f}  KL {kl[a:b].mean():.4f}")
