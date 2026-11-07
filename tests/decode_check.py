"""The engine's token-by-token decode against a full recompute: greedy continuation with xtg-cli generate, then
the reference (reference/dory_ref.py) runs prompt + continuation in one pass; at every generated position its
argmax must be the token the engine chose (near-ties aside: reported with the margin).

    python tests/decode_check.py --cli build/xtg-cli --model DIR --ids PROMPT.json [--max 48] [--opt k=v ...]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference'))
from dory_ref import Ref  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--max', type=int, default=48)
    p.add_argument('--opt', action='append', default=[])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    prompt = json.load(open(a.ids))
    cmd = [a.cli, 'generate', '--model', a.model, '--ids', a.ids, '--max', str(a.max)]
    for o in a.opt:
        cmd += ['--opt', o]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    gen = out['ids']
    print(f"engine: {len(gen)} tokens, decode {out['decode_tok_s']:.1f} tok/s: {out['text'][:120]!r}")
    opts = dict(o.split('=', 1) for o in a.opt)
    m = Ref(a.model, int(opts['loops']) if 'loops' in opts else None, opts.get('kv_mode'), device=a.device)
    logits = m.forward(prompt + gen[:-1])
    rows = logits[len(prompt) - 1:]
    bad = 0
    for i, tok in enumerate(gen):
        top = rows[i].topk(2)
        if int(top.indices[0]) != tok:
            margin = float(top.values[0] - rows[i][tok])
            print(f'  position {i}: engine {tok}, reference {int(top.indices[0])} (logit margin {margin:.3f})')
            bad += margin > 0.05
    print(f'{len(gen) - bad}/{len(gen)} generated tokens are the reference argmax (ties within 0.05 forgiven)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
