"""kv_enc_check.py PTE [OUT.kvmap.json]: do the KV outputs of each layer carry the same quantization encoding as that
layer's cache input? libxtg_npu writes the output bytes back into the caches. Per shard (QNN context), the cache
inputs args_i (K: i < n, V: n + l) and the 8-bit outputs of K ([.., hd, rows]) and V ([.., rows, hd]) in graph order;
prints the pairs that differ. In these exports every V cache carries the encoding of the next layer's new V
(ExecuTorch's KV annotation); with OUT, the byte maps libxtg_npu applies when it writes the caches (option
kv_requant, by default NAME.kvmap.json next to NAME.pte). Needs QNN_SDK_ROOT and ExecuTorch's Python environment."""
import json
import os
import re
import subprocess
import sys

from executorch.backends.qualcomm.utils.utils import dump_context_from_pte

U = os.path.join(os.environ["QNN_SDK_ROOT"], "bin/x86_64-linux-clang/qnn-context-binary-utility")
pte = os.path.abspath(sys.argv[1])
files = sorted((f for f in dump_context_from_pte(pte) if os.path.basename(f).startswith("kv_forward")),
               key=lambda f: int(re.search(r"_(\d+)\.bin$", f).group(1)))


def walk(o, out):
    if isinstance(o, dict):
        if "name" in o and "quantizeParams" in o and "dimensions" in o:
            out.append(o)
        for v in o.values():
            walk(v, out)
    elif isinstance(o, list):
        for v in o:
            walk(v, out)


def enc(t):
    q = t["quantizeParams"].get("scaleOffset", {})
    return round(q.get("scale", 0), 9), q.get("offset")


bad = total = 0
all_k_out, all_v_out, all_in = [], [], {}
for f in files:
    subprocess.run([U, "--context_binary", f, "--json_file", f + ".json"], check=True, capture_output=True)
    ts, seen = [], set()
    walk(json.load(open(f + ".json")), ts)
    ts = [t for t in ts if not (t["name"] in seen or seen.add(t["name"]))]
    ins = {int(m.group(1)): t for t in ts if (m := re.match(r"input_\d+_args_(\d+)@", t["name"]))}
    outs = [t for t in ts if t["name"].startswith("output_") and "UFIXED_POINT_8" in t.get("dataType", "")]
    num = lambda t: int(re.search(r"permute_copy_default_(\d+)", t["name"]).group(1))
    outs.sort(key=num)
    k_out = [t for t in outs if t["dimensions"][-1] != 256]   # [.., hd, rows]
    v_out = [t for t in outs if t["dimensions"][-1] == 256]   # [.., rows, hd]
    all_k_out += [(num(t), enc(t)) for t in k_out]
    all_v_out += [(num(t), enc(t)) for t in v_out]
    all_in.update({i: enc(t) for i, t in ins.items()})
    keys = sorted(ins)
    half = len(keys) // 2
    k_in, v_in = [ins[i] for i in keys[:half]], [ins[i] for i in keys[half:]]
    print(f"{os.path.basename(f)}: {len(k_in)} K / {len(v_in)} V cache inputs, {len(k_out)} K / {len(v_out)} V outputs")
    for kind, a, b in (("K", k_out, k_in), ("V", v_out, v_in)):
        for l, (o, i) in enumerate(zip(a, b)):
            total += 1
            if enc(o) != enc(i):
                bad += 1
                print(f"  {kind} layer {l}: output {enc(o)} {o['name'][:44]}, cache {enc(i)} {i['name'][:24]}")
print(f"{bad} of {total} pairs differ")
if len(sys.argv) > 2:   # the byte maps for libxtg_npu's kv_requant: [s_out, zp_out, s_in, zp_in] for K_0.., then V_0..
    n = len(all_in) // 2
    k_out = [e for _, e in sorted(all_k_out)]
    v_out = [e for _, e in sorted(all_v_out)]
    assert len(k_out) == len(v_out) == n, (len(k_out), len(v_out), n)
    rows = [[o[0], -o[1], all_in[l][0], -all_in[l][1]] for l, o in enumerate(k_out)]
    rows += [[o[0], -o[1], all_in[n + l][0], -all_in[n + l][1]] for l, o in enumerate(v_out)]
    json.dump(rows, open(sys.argv[2], "w"))
    changed = sum(r[0] != r[2] or r[1] != r[3] for r in rows)
    print(f"wrote {sys.argv[2]}: {len(rows)} instances, {changed} need a byte map")
