"""Compare two runs of the same prompt (xtg-cli forward, reference/dory_ref.py, reference/vllm_logprobs.py).

    python tests/forward_check.py A.json B.json [--logits A.f32 B.f32 --vocab 131072] [--max-mean 0.02]

Prints the mean and largest difference of the next-token log probabilities, how often the argmax agrees, the
NLL of both, and with --logits the largest logit difference. Exit 1 when the mean difference is above --max-mean
or the argmax agrees less than --min-agree of the time.
"""
import argparse
import json
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--logits', nargs=2)
    p.add_argument('--vocab', type=int, default=131072)
    p.add_argument('--max-mean', type=float, default=0.02)
    p.add_argument('--min-agree', type=float, default=0.97)
    p.add_argument('--skip', type=int, default=0, help='positions at the start to leave out')
    a = p.parse_args()
    A, B = json.load(open(a.a)), json.load(open(a.b))
    la = [x for x in A['logprob_next']]
    lb = [x for x in B['logprob_next']]
    n = min(len(la), len(lb))
    pairs = [(x, y) for x, y in zip(la[a.skip:n], lb[a.skip:n]) if x is not None and y is not None]
    d = np.abs(np.array([x - y for x, y in pairs]))
    agree = np.mean([x == y for x, y in zip(A['argmax'][a.skip:n], B['argmax'][a.skip:n])])
    nll_a, nll_b = -np.mean([x for x, _ in pairs]), -np.mean([y for _, y in pairs])
    print(f'{len(pairs)} positions: |dlogprob| mean {d.mean():.4f} p99 {np.percentile(d, 99):.4f} max {d.max():.4f}; '
          f'argmax agree {agree:.4f}; NLL {nll_a:.4f} vs {nll_b:.4f}')
    if a.logits:
        x = np.fromfile(a.logits[0], dtype=np.float32).reshape(-1, a.vocab)
        y = np.fromfile(a.logits[1], dtype=np.float32).reshape(-1, a.vocab)
        r = min(len(x), len(y))
        print(f'logits: max |diff| {np.abs(x[:r] - y[:r]).max():.5f}, mean {np.abs(x[:r] - y[:r]).mean():.6f}')
    ok = d.mean() <= a.max_mean and agree >= a.min_agree
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
