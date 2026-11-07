"""Real Dory activations for the NPU probe (server side, the FP32 PyTorch reference).

    python tools/npu_probe/capture.py --model DIR --out DIR [--layer 32] [--device cuda]

Runs reference/dory_ref.py over two texts made from this repository (calibration: the docs; test: the C++ sources
and the plan), 4096 tokens each, and saves for attention sublayer L in the last loop its input x [T, 2560], the
attention output before the output gate att [T, 4096] and the g_proj output gate [T, 4096] (positions 1024..3071),
as calib.npz / test.npz (FP32). The NPU segment starts there: gate, o_proj, slerp, the MLP L+1, slerp, the q / k /
v / g projections of L+2 (tools/npu_probe/make_models.py).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'reference'))
from dory_ref import Ref  # noqa: E402


class Capture(Ref):
    def __init__(self, *a, layer, **k):
        super().__init__(*a, **k)
        self.layer = layer
        self.calls = 0
        self.saved = None

    def _sublayer(self, li, x, pos, inst, store):
        if li != self.layer:
            return super()._sublayer(li, x, pos, inst, store)
        self.calls += 1
        L = self.layers[li]
        t = x.shape[0]
        q = self.mm(x, L['q_proj']).view(t, self.nh, self.hd)
        k = self.mm(x, L['k_proj']).view(t, self.nkv, self.hd)
        v = self.mm(x, L['v_proj']).view(t, self.nkv, self.hd)
        gate = self.mm(x, L['g_proj'])
        q, k = self.rope(q, pos, L['inv_freq']), self.rope(k, pos, L['inv_freq'])
        q = (L['qs'].view(1, self.nh, self.hd) * self.justnorm(q).float()).to(self.dt)
        k = self.justnorm(k)
        att = self.attention(li, inst, q, k, v, int(pos[0]), store).reshape(t, -1)
        self.saved = (x.float().cpu().numpy(), att.float().cpu().numpy(), gate.float().cpu().numpy())   # the last loop wins
        out = (att.float() * torch.sigmoid(gate.float() * L['gs'])).to(self.dt)
        out = self.justnorm(out)
        return self.slerp(x, self.mm(out, L['o_proj']), L['alpha'])


def text(patterns):
    parts = []
    for p in patterns:
        for f in sorted(glob.glob(os.path.join(ROOT, p))):
            parts.append(open(f, encoding='utf-8', errors='ignore').read())
    return '\n\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layer', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tokens', type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(a.model, 'tokenizer.json'))
    m = Capture(a.model, device=a.device, layer=a.layer)
    assert m.types[a.layer] != 'mlp' and m.types[a.layer + 1] == 'mlp' and m.types[a.layer + 2] != 'mlp'
    sources = {'calib': ['docs/*.md', 'docs/reports/*.md', 'README.md'],
               'test': ['engine/src/*.cpp', 'packs/dory/*.cpp', 'server/*.cpp', 'docs/PLAN.md']}
    meta = {'layer': a.layer, 'loops': m.loops, 'rows': [1024, 3072]}
    for name, pats in sources.items():
        ids = tok.encode(text(pats)).ids[:a.tokens]
        assert len(ids) == a.tokens, f'{name}: only {len(ids)} tokens'
        m.reset()
        m.saved = None
        m.calls = 0
        m.forward(ids)
        x, att, gate = (v[1024:3072] for v in m.saved)
        np.savez(os.path.join(a.out, f'{name}.npz'), x=x, att=att, gate=gate)
        meta[name] = {'calls': m.calls, 'x_norm_mean': float(np.linalg.norm(x, axis=1).mean())}
        print(name, x.shape, att.shape, gate.shape, meta[name], flush=True)
    json.dump(meta, open(os.path.join(a.out, 'capture.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
