"""hf_local_variant.py HF_HOME NAME BF16_DIR CONFIG.json: an offline Hugging Face repo local/NAME for a Dory variant
(dory-tiny, dory-4b-l1): its own config.json, every other file of the BF16 checkpoint linked. The export reads it
with HF_HUB_OFFLINE=1 and caches the converted weights under the repo name, apart from the full model's.
"""
import os
import shutil
import sys

home, name, src, config = sys.argv[1:5]
root = os.path.join(home, "hub", f"models--local--{name}")
snap = os.path.join(root, "snapshots", name)
os.makedirs(os.path.join(root, "refs"), exist_ok=True)
os.makedirs(snap, exist_ok=True)
open(os.path.join(root, "refs", "main"), "w").write(name)
for f in os.listdir(src):
    dst = os.path.join(snap, f)
    if os.path.lexists(dst):
        os.remove(dst)
    if f == "config.json":
        shutil.copyfile(config, dst)
    elif os.path.isfile(os.path.join(src, f)):
        os.symlink(os.path.abspath(os.path.join(src, f)), dst)
print(f"local/{name}: {len(os.listdir(snap))} files in {snap}")
