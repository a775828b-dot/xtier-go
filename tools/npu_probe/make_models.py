"""ONNX models, inputs and references for the NPU probe (server side; numpy + onnx, no torch).

    python tools/npu_probe/make_models.py --model DIR --capture DIR --out DIR [--layer 32] [--m 512,768,1024] [--plain]

The segment is everything between two attention calls of Dory, with the real NVFP4 weights decoded exactly to
FP32: the output gate and norm of attention sublayer L, its o_proj, slerp, the MLP L+1 (gate / up as one matrix,
SiLU with the suv scales, norm, down), slerp, then the q / k / v / g projections of attention sublayer L+2
(91.75 M weights, 2 x 91.75 M FLOP per token). Inputs x [M, 2560], att [M, 4096], gate [M, 4096]; outputs
h2 [M, 2560], q [M, 4096], k [M, 1024], v [M, 1024], g2 [M, 4096]; M <= 2048 (the test window). The graph is
folded by default (segment_onnx; 2.8x faster on the Y900, same error), --plain keeps every norm and scale. Also
tiny_m16: one 2560 x 2560 matrix for the fixed cost of an execute. Writes onnx/*.onnx, raw float32 inputs, the input lists (calib_m*.txt: absolute paths
for the quantizer; list_m*.txt: relative to the kit for qnn-net-run), ref.npz (float64 math on the test rows) and
manifest.json.
"""
import argparse
import json
import os
import struct

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        e, m = (b >> 3) & 15, b & 7
        v = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        t[b] = -v if b & 128 else v
    t[0x7f] = t[0xff] = np.nan
    return t


class Safetensors:
    """Reads tensors straight from the file (numpy has no FP8 / BF16)."""

    def __init__(self, path):
        self.f = open(path, 'rb')
        n = struct.unpack('<Q', self.f.read(8))[0]
        self.meta = json.loads(self.f.read(n))
        self.base = 8 + n
        self.e4m3 = e4m3_table()

    def get(self, name):
        m = self.meta[name]
        a, b = m['data_offsets']
        self.f.seek(self.base + a)
        buf, dt, shape = self.f.read(b - a), m['dtype'], m['shape']
        if dt == 'F32':
            return np.frombuffer(buf, np.float32).reshape(shape).copy()
        if dt == 'BF16':
            return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == 'F8_E4M3':
            return self.e4m3[np.frombuffer(buf, np.uint8)].reshape(shape)
        if dt == 'U8':
            return np.frombuffer(buf, np.uint8).reshape(shape)
        raise ValueError(f'{name}: {dt}')

    def nvfp4(self, prefix):
        w = self.get(prefix + '.weight')                       # [n, k/2], low nibble first
        s = self.get(prefix + '.weight_scale')                 # [n, k/16]
        g = float(self.get(prefix + '.weight_scale_2').reshape(-1)[0])
        codes = np.stack([w & 15, w >> 4], -1).reshape(w.shape[0], -1)
        return E2M1[codes] * np.repeat(s, 16, axis=1) * np.float32(g)


def weights(st, c, L):
    std = c['initializer_range']
    alpha_scale = c['ngpt_alpha_init'] / std
    g0 = c.get('attention_output_gate_logit_scale_init') or c['hidden_size'] ** 0.5
    gate_factor, suv_factor = g0 ** 0.5 / std, c['hidden_size'] ** 0.5
    p = lambda i: f'backbone.layers.{i}.'
    W = {
        'gs': (st.get(p(L) + 'mixer.s_gate') * gate_factor) ** 2,
        'wo': st.nvfp4(p(L) + 'mixer.o_proj'),
        'lr1': np.abs(st.get(p(L) + 'attn_alpha') * alpha_scale),
        'wgu': np.concatenate([st.nvfp4(p(L + 1) + 'mixer.gate_proj'), st.nvfp4(p(L + 1) + 'mixer.up_proj')]),
        'sg': st.get(p(L + 1) + 'mixer.suv_gate') * suv_factor,
        'su': st.get(p(L + 1) + 'mixer.suv_up') * suv_factor,
        'wd': st.nvfp4(p(L + 1) + 'mixer.down_proj'),
        'lr2': np.abs(st.get(p(L + 1) + 'mlp_alpha') * alpha_scale),
        'wqkvg': np.concatenate([st.nvfp4(p(L + 2) + f'mixer.{n}_proj') for n in ('q', 'k', 'v', 'g')]),
    }
    return {k: np.ascontiguousarray(v, np.float32) for k, v in W.items()}


def reference(W, x, att, gate):
    """The segment in float64: the truth every NPU variant is compared with."""
    f = lambda a: a.astype(np.float64)
    norm = lambda a: a / np.sqrt((a * a).sum(-1, keepdims=True) + 1e-6)
    slerp = lambda a, b, lr: norm(norm(a) + lr * (norm(b) - norm(a)))
    a = f(att) * (1 / (1 + np.exp(-f(gate) * f(W['gs']))))
    h = slerp(f(x), norm(a) @ f(W['wo']).T, f(W['lr1']))
    gu = h @ f(W['wgu']).T
    F = gu.shape[1] // 2
    g, u = gu[:, :F] * f(W['sg']), gu[:, F:] * f(W['su'])
    m = norm(g / (1 + np.exp(-g)) * u)
    h2 = slerp(h, m @ f(W['wd']).T, f(W['lr2']))
    q, k, v, g2 = np.split(h2 @ f(W['wqkvg']).T, [4096, 5120, 6144], axis=1)
    return {n: t.astype(np.float32) for n, t in zip(('h2', 'q', 'k', 'v', 'g2'), (h2, q, k, v, g2))}


def save(model, path):
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)


def segment_onnx(W, M, path, fold=False):
    # fold: the same math with fewer [M, C] passes. norm(c * v) = norm(v) for a row scalar c > 0, so the norms of
    # att_g and m go (o and down are normalized by the slerp anyway), the slerp inputs x and h are already unit
    # rows, and the SwiGLU scales move into the gate / up rows (a per-row weight scale absorbs them exactly)
    if fold:
        W = dict(W, wgu=W['wgu'] * np.concatenate([W['sg'], W['su']])[:, None])
        del W['sg'], W['su']
    nodes = []
    init = [numpy_helper.from_array(v, k) for k, v in W.items()]
    init += [numpy_helper.from_array(np.array(s, np.int64), name) for name, s in
             (('split_gu', [7168, 7168]), ('split_qkvg', [4096, 1024, 1024, 4096]), ('axis1', [1]))]
    init.append(numpy_helper.from_array(np.array([1e-6], np.float32), 'eps'))

    def op(kind, ins, out, **attrs):
        nodes.append(helper.make_node(kind, ins, [out], name=out, **attrs))
        return out

    # justnorm from elementwise ops and a row sum: LpNormalization becomes one HTP L2Norm that needs the whole
    # [M, C] tensor in VTCM (8 MiB) and fails to prepare at M = 1024 with 16-bit activations; a [M, C] / [M, 1]
    # Div made the M = 1024 segment 1.3-2x slower on the Y900 than one reciprocal per row and a Mul
    def l2(a, out):
        ss = op('ReduceSum', [op('Mul', [a, a], out + '_sq'), 'axis1'], out + '_ss', keepdims=1)
        inv = op('Reciprocal', [op('Sqrt', [op('Add', [ss, 'eps'], out + '_se')], out + '_rn')], out + '_inv')
        return op('Mul', [a, inv], out)

    def slerp(a, b, lr, out):
        ya, yb = a if fold else l2(a, out + '_ya'), l2(b, out + '_yb')
        d = op('Mul', [op('Sub', [yb, ya], out + '_d'), lr], out + '_e')
        return l2(op('Add', [ya, d], out + '_f'), out)

    g = op('Sigmoid', [op('Mul', ['gate', 'gs'], 'gate_s')], 'gate_sig')
    a = op('Mul', ['att', g], 'att_g')
    h = slerp('x', op('Gemm', [a if fold else l2(a, 'att_n'), 'wo'], 'o', transB=1), 'lr1', 'h')
    gu = op('Gemm', [h, 'wgu'], 'gu', transB=1)
    nodes.append(helper.make_node('Split', [gu, 'split_gu'], ['mg', 'mu'], name='split_gu_n', axis=1))
    gs, us = ('mg', 'mu') if fold else (op('Mul', ['mg', 'sg'], 'mg_s'), op('Mul', ['mu', 'su'], 'mu_s'))
    m = op('Mul', [op('Mul', [gs, op('Sigmoid', [gs], 'mg_sig')], 'silu'), us], 'm')
    d = op('Gemm', [m if fold else l2(m, 'm_n'), 'wd'], 'down', transB=1)
    slerp(h, d, 'lr2', 'h2')
    qkvg = op('Gemm', ['h2', 'wqkvg'], 'qkvg', transB=1)
    nodes.append(helper.make_node('Split', [qkvg, 'split_qkvg'], ['q', 'k', 'v', 'g2'], name='split_qkvg_n', axis=1))
    vi = lambda name, c: helper.make_tensor_value_info(name, TensorProto.FLOAT, [M, c])
    graph = helper.make_graph(nodes, f'seg_m{M}', [vi('x', 2560), vi('att', 4096), vi('gate', 4096)],
                              [vi('h2', 2560), vi('q', 4096), vi('k', 1024), vi('v', 1024), vi('g2', 4096)], init)
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], producer_name='xtier-go npu probe'), path)


def tiny_onnx(W, M, path):
    w = np.ascontiguousarray(W['wqkvg'][:2560])
    graph = helper.make_graph([helper.make_node('Gemm', ['x', 'w'], ['y'], name='y', transB=1)], f'tiny_m{M}',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [M, 2560])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [M, 2560])],
                              [numpy_helper.from_array(w, 'w')])
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], producer_name='xtier-go npu probe'), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--capture', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layer', type=int, default=32)
    ap.add_argument('--m', default='512,768,1024')
    ap.add_argument('--plain', action='store_true', help='every norm and scale as written (default: folded)')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for d in ('onnx', 'inputs', 'calib'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    c = json.load(open(os.path.join(a.model, 'config.json')))
    W = weights(Safetensors(os.path.join(a.model, 'model.safetensors')), c, a.layer)
    calib, test = np.load(os.path.join(a.capture, 'calib.npz')), np.load(os.path.join(a.capture, 'test.npz'))
    np.savez(os.path.join(out, 'ref.npz'), **reference(W, test['x'], test['att'], test['gate']))
    params = int(sum(W[k].size for k in ('wo', 'wgu', 'wd', 'wqkvg')))
    manifest = {'layer': a.layer, 'params': params, 'models': {}}
    for M in [int(s) for s in a.m.split(',')] + [16]:
        tiny = M == 16
        name = 'tiny_m16' if tiny else f'seg_m{M}'
        if tiny:
            tiny_onnx(W, M, os.path.join(out, 'onnx', name + '.onnx'))
        else:
            segment_onnx(W, M, os.path.join(out, 'onnx', name + '.onnx'), not a.plain)
        ins = ('x',) if tiny else ('x', 'att', 'gate')
        for t in ins:   # the test input: the first M rows of the test window
            test[t][:M].astype(np.float32).tofile(os.path.join(out, 'inputs', f'{t}_m{M}.raw'))
        with open(os.path.join(out, f'list_m{M}.txt'), 'w') as fd:
            fd.write(' '.join(f'{t}:=inputs/{t}_m{M}.raw' for t in ins) + '\n')
        with open(os.path.join(out, f'calib_m{M}.txt'), 'w') as fd:
            for b in range(max(1, min(4, 2048 // M))):
                row = []
                for t in ins:
                    p = os.path.join(out, 'calib', f'{t}_m{M}_{b}.raw')
                    calib[t][b * M:(b + 1) * M].astype(np.float32).tofile(p)
                    row.append(f'{t}:={p}')
                fd.write(' '.join(row) + '\n')
        manifest['models'][name] = {'M': M, 'flop': 2 * (2560 * 2560 if tiny else params) * M, 'inputs': list(ins)}
        print(name, 'ok', flush=True)
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
