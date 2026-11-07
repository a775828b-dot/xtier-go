"""INT4 weight encodings with 16-element blocks (the NVFP4 block) for the four segment matrices.

    python tools/npu_probe/make_overrides.py --model DIR --out DIR [--layer 32] [--block 16]

bq<B>.json: blockwise INT4 (one FP32 scale per block, symmetric, the block's max over 7); lpbq<B>.json: low power
blockwise (one FP32 scale per row times a 4-bit integer per block, what Qualcomm's LLMs use on HTP); bq8x16.json:
INT8 with half the NVFP4 block scale, which holds NVFP4 exactly. QAIRT 2.0.0 overrides with the weights only, so
qairt-quantizer still calibrates the activations. Prints the weight error of each against the exact NVFP4 values.
"""
import argparse
import json
import os

import numpy as np

from make_models import Safetensors, weights


def bq(w, block):
    n, k = w.shape
    m = np.abs(w.reshape(n, k // block, block)).max(-1)
    scale = np.maximum(m / 7, 1e-12)
    q = np.clip(np.round(w.reshape(n, k // block, block) / scale[..., None]), -8, 7)
    return scale, (q * scale[..., None]).reshape(n, k)


def lpbq(w, block):
    n, k = w.shape
    m = np.abs(w.reshape(n, k // block, block)).max(-1) / 7         # the scale each block wants
    ch = np.maximum(m.max(-1) / 16, 1e-12)                            # per row: 16 integer steps cover the largest
    ib = np.clip(np.ceil(m / ch[:, None]), 1, 16)
    scale = ib * ch[:, None]
    q = np.clip(np.round(w.reshape(n, k // block, block) / scale[..., None]), -8, 7)
    return ch, ib.astype(int), (q * scale[..., None]).reshape(n, k)


def nvfp4_scales(st, L):
    """Per-16 block scales of the four matrices, rows in the order of weights(): E4M3 scale x tensor scale."""
    p = lambda i: f'backbone.layers.{i}.mixer.'
    parts = {'wo': [p(L) + 'o_proj'], 'wgu': [p(L + 1) + 'gate_proj', p(L + 1) + 'up_proj'], 'wd': [p(L + 1) + 'down_proj'],
             'wqkvg': [p(L + 2) + f'{n}_proj' for n in ('q', 'k', 'v', 'g')]}
    return {k: np.concatenate([st.get(n + '.weight_scale') * float(st.get(n + '.weight_scale_2').reshape(-1)[0]) for n in v])
            for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layer', type=int, default=32)
    ap.add_argument('--block', type=int, default=16)
    a = ap.parse_args()
    c = json.load(open(os.path.join(a.model, 'config.json')))
    st = Safetensors(os.path.join(a.model, 'model.safetensors'))
    W = weights(st, c, a.layer)
    S = nvfp4_scales(st, a.layer)
    B = a.block
    enc_bq, enc_lp, enc_b8 = [], [], []
    for name in ('wo', 'wgu', 'wd', 'wqkvg'):
        w = W[name].astype(np.float64)
        s, dq = bq(w, B)
        ch, ib, dl = lpbq(w, B)
        # INT8 with half the NVFP4 block scale holds every E2M1 value exactly (codes x 2: 0, 1, 2, 3, 4, 6, 8, 12)
        s8 = S[name].astype(np.float64) / 2
        q8 = np.round(w.reshape(w.shape[0], -1, 16) / np.maximum(s8, 1e-30)[..., None])
        d8 = (q8 * s8[..., None]).reshape(w.shape)
        rel = lambda d: float(np.sqrt(((d - w) ** 2).sum() / (w ** 2).sum()))
        print(f'{name} {w.shape}: BQ{B} weight error {rel(dq) * 100:.2f}%, LPBQ{B} {rel(dl) * 100:.2f}%, '
              f'INT8 BQ16 {rel(d8) * 100:.4f}% (codes {int(q8.min())}..{int(q8.max())})')
        # no y_zero_point (symmetric: QAIRT 2.47 cannot read a nested zero-point list); LPBQ wants the row scales
        # with the rank of the block scales and the block axis to infer the channel axis
        enc_bq.append({'name': name, 'output_dtype': 'int4', 'y_scale': s.tolist(), 'axis': 1, 'block_size': B})
        enc_lp.append({'name': name, 'output_dtype': 'int4', 'per_channel_float_scale': ch[:, None].tolist(),
                       'per_block_int_scale': ib.tolist(), 'axis': 1, 'block_size': B})
        enc_b8.append({'name': name, 'output_dtype': 'int8', 'y_scale': s8.tolist(), 'axis': 1, 'block_size': 16})
    os.makedirs(a.out, exist_ok=True)
    for fname, enc in ((f'bq{B}.json', enc_bq), (f'lpbq{B}.json', enc_lp), ('bq8x16.json', enc_b8)):
        # 2.0.0: one list of named tensor encodings (only the weights here: the quantizer calibrates the rest)
        json.dump({'version': '2.0.0', 'encodings': enc}, open(os.path.join(a.out, fname), 'w'))


if __name__ == '__main__':
    main()
