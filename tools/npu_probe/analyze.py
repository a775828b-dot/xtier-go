"""NPU probe results (server side).

    python tools/npu_probe/analyze.py emu WORK            HTP x86 emulator: run every seg_m512 DLC, compare with ref.npz
    python tools/npu_probe/analyze.py device WORK RESULTS  results/ from run_probe.sh: times, profiles, first rows

Accuracy per output (h2, q, k, v, g2): relative RMS error against the float64 reference and the cosine. Speed:
the average execute time of qnn-profile-viewer ("NetRun" per inference, else the wall time per inference),
TFLOP/s from manifest.json, and what it means for Dory prefill: 56 segments per token at 2 loops, so linear-only
tokens/s = M / (56 x time). Prints a markdown table; --json writes the numbers too.
"""
import argparse
import glob
import json
import os
import re
import subprocess

import numpy as np

WIDTH = {'h2': 2560, 'q': 4096, 'k': 1024, 'v': 1024, 'g2': 4096, 'y': 2560}
SEGMENTS = 56   # block evaluations per token at 2 loops (8 + 20 x 2 + 8)


def errors(out_dir, ref, rows):
    r = {}
    for name in ('h2', 'q', 'k', 'v', 'g2'):
        p = os.path.join(out_dir, name + '.raw')
        if not os.path.exists(p):
            continue
        y = np.fromfile(p, np.float32).reshape(-1, WIDTH[name])[:rows].astype(np.float64)
        t = ref[name][:len(y)].astype(np.float64)
        r[name] = {'rel_rms': float(np.sqrt(((y - t) ** 2).sum() / (t ** 2).sum())),
                   'cos': float((y * t).sum() / np.linalg.norm(y) / np.linalg.norm(t))}
    return r


def emu(work):
    """seg_m512 DLCs in the HTP x86 emulator as v79 (a float graph needs the SoC: the emulator then uses FP16)."""
    work = os.path.abspath(work)
    lib = os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'x86_64-linux-clang')
    os.makedirs(os.path.join(work, 'ctx'), exist_ok=True)
    v79 = os.path.join(work, 'ctx', 'emu_v79.json')
    json.dump({'graphs': [{'vtcm_mb': 8, 'O': 3, 'graph_names': ['seg_m512']}],
               'devices': [{'dsp_arch': 'v79', 'soc_model': 69, 'pd_session': 'unsigned', 'device_id': 0}]}, open(v79, 'w'))
    config = os.path.join(work, 'ctx', 'emu.json')
    json.dump({'backend_extensions': {'shared_library_path': os.path.join(lib, 'libQnnHtpNetRunExtensions.so'),
                                      'config_file_path': v79}}, open(config, 'w'))
    ref = np.load(os.path.join(work, 'ref.npz'))
    out = {}
    for dlc in sorted(glob.glob(os.path.join(work, 'dlc', 'seg_m512*.dlc'))):
        name = os.path.basename(dlc)[:-4]
        variant = name.split('_', 2)[2] if name.count('_') >= 2 else 'fp16'
        res = os.path.join(work, 'emu', name)
        if not os.path.exists(os.path.join(res, 'Result_0', 'h2.raw')):
            subprocess.run([os.path.join(os.environ['QNN_SDK_ROOT'], 'bin', 'x86_64-linux-clang', 'qnn-net-run'),
                            '--backend', os.path.join(lib, 'libQnnHtp.so'), '--model', os.path.join(lib, 'libQnnModelDlc.so'),
                            '--dlc_path', dlc, '--input_list', 'list_m512.txt', '--output_dir', res, '--config_file', config],
                           cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out[variant] = errors(os.path.join(res, 'Result_0'), ref, 512)
    print('| 精度 | ' + ' | '.join(f'{n} 相对误差' for n in ('h2', 'q', 'k', 'v', 'g2')) + ' | h2 余弦 |')
    print('|---|' + '---:|' * 6)
    for v, e in out.items():
        print(f'| {v} | ' + ' | '.join(f"{e[n]['rel_rms'] * 100:.2f}%" for n in ('h2', 'q', 'k', 'v', 'g2')) +
              f" | {e['h2']['cos']:.6f} |")
    return out


def profile_avg_us(text):
    """Average per-inference execute time from qnn-profile-viewer output (microseconds), or None."""
    for key in (r'NetRun:\s*([\d.]+)\s*us', r'Execute\s*\(?.*?\)?\s*.*?:\s*([\d.]+)\s*us'):
        m = re.findall(key, text)
        if m:
            return float(m[-1])
    return None


def device(work, results):
    manifest = json.load(open(os.path.join(work, 'manifest.json')))['models']
    ref = np.load(os.path.join(work, 'ref.npz'))
    rows = []
    for line in open(os.path.join(results, 'times.txt')):
        f = line.split()
        name = f[0]
        kv = dict(x.split('=', 1) for x in f[1:] if '=' in x)
        graph = '_'.join(name.split('_')[:2])
        if graph not in manifest:
            continue
        prof = os.path.join(results, 'prof', name + '.profile.txt')
        avg = profile_avg_us(open(prof).read()) if os.path.exists(prof) else None
        n = int(kv.get('n', 1))
        t = avg / 1e6 if avg else float(kv.get('wall_s', 'nan')) / n
        M, flop = manifest[graph]['M'], manifest[graph]['flop']
        acc = errors(os.path.join(results, 'acc', name), ref, 32) if graph.startswith('seg') else {}
        rows.append({'name': name, 'M': M, 'ms': t * 1e3, 'tflops': flop / t / 1e12,
                     'dory_linear_tok_s': M / (SEGMENTS * t) if graph.startswith('seg') else None,
                     'rc': kv.get('rc'), 'acc': acc, 'source': 'profile' if avg else 'wall'})
    print('| 模型 | M | 每次 execute | TFLOP/s | Dory 线性层 tok/s | h2 误差 | k 误差 | 计时 |')
    print('|---|---:|---:|---:|---:|---:|---:|---|')
    for r in rows:
        e = r['acc']
        print(f"| {r['name']} | {r['M']} | {r['ms']:.2f} ms | {r['tflops']:.2f} | "
              f"{r['dory_linear_tok_s'] or 0:.0f} | {e.get('h2', {}).get('rel_rms', float('nan')) * 100:.2f}% | "
              f"{e.get('k', {}).get('rel_rms', float('nan')) * 100:.2f}% | {r['source']} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['emu', 'device'])
    ap.add_argument('work')
    ap.add_argument('results', nargs='?')
    ap.add_argument('--json')
    a = ap.parse_args()
    out = emu(a.work) if a.mode == 'emu' else device(a.work, a.results)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
