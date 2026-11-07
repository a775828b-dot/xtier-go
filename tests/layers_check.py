"""Compare two directories of sublayer outputs (reference/vllm_layers.py, reference/dory_ref.py --dump).

    python tests/layers_check.py DIR_A DIR_B

Prints, in execution order, each sublayer call's relative error (|a - b| / |b|) and cosine similarity, so the
first call that departs shows where two implementations differ.
"""
import re
import sys

import torch


def order(names):
    def key(n):
        m = re.match(r'layer(\d+)_call(\d+)\.pt', n)
        i, k = int(m.group(1)), int(m.group(2))
        # execution order: input layers, then recurrent layers loop by loop, then output layers
        return (k, i) if 16 <= i < 56 else ((-1, i) if i < 16 else (99, i))
    return sorted(names, key=key)


def main():
    import os
    a, b = sys.argv[1], sys.argv[2]
    names = order([n for n in os.listdir(a) if n.startswith('layer') and os.path.exists(os.path.join(b, n))])
    for n in names:
        x, y = torch.load(os.path.join(a, n)), torch.load(os.path.join(b, n))
        rel = ((x - y).norm() / y.norm()).item()
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
        worst = (x - y).norm(dim=-1).argmax().item()
        print(f'{n:20s} rel {rel:.5f} cos {cos:.6f} worst token {worst}')


if __name__ == '__main__':
    main()
