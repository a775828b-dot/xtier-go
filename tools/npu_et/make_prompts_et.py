"""make_prompts_et.py OUT_DIR: tokenized prompts (raw uint64, what qnn_llama_runner --tokenized_prompt reads) for the
NPU spike, from the S1 agent traces.

  dory-agentN.bin   a whole agent prompt of the 3k session (Dory ids, ends with the generation prompt), N tokens
  dory-tailN.bin    the last N tokens of the 28k session's first prompt (for speed at a given length)
  qwen-agentN.bin   the same 3k prompt as text, re-rendered with Qwen3's chat template (Qwen ids)
"""
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
recs = [json.loads(l) for l in open(os.environ["DORY_TRACES"] + "/s1-think.jsonl")]
dtok = Tokenizer.from_file(os.environ["DORY_TOKENIZER"])


def save(name, ids):
    np.asarray(ids, dtype=np.uint64).tofile(os.path.join(out, name))
    print(name, len(ids))


r3k = next(r for r in recs if r["session"] == "3k")
save(f"dory-agent{len(r3k['prompt'])}.bin", r3k["prompt"])
r28 = next(r for r in recs if r["session"] == "28k")
for n in (512, 1024, 2048, 3072):
    save(f"dory-tail{n}.bin", r28["prompt"][-n:])
# Qwen: the 3k prompt's text in Qwen3's template
from transformers import AutoTokenizer  # noqa: E402

qt = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
text = dtok.decode(r3k["prompt"], skip_special_tokens=False)
# Dory's turns are ChatML too: <|im_start|>role\n ... <|im_end|>; keep the text and let Qwen's tokenizer split it
q = qt(text, add_special_tokens=False)["input_ids"]
save(f"qwen-agent{len(q)}.bin", q)
qtail = qt(dtok.decode(r28["prompt"], skip_special_tokens=False), add_special_tokens=False)["input_ids"]
for n in (512, 1024, 2048, 3072):
    save(f"qwen-tail{n}.bin", qtail[-n:])
