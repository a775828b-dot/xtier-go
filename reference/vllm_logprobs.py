"""Prompt log probabilities from vLLM (the reference implementation of Dory), in xtg-cli forward's JSON.

    python reference/vllm_logprobs.py --model DIR --ids A.json [B.json ...] --out-dir OUT [--loops 2]
                                      [--kv-mode per_loop] [--chunk 8192]

Writes OUT/<name>.json per ids file: the next token's log probability and the argmax at every position. Needs
vLLM with Dory (main after PR #60149, or 0.33.0) and a GPU it supports.
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--loops', type=int, default=2)
    p.add_argument('--kv-mode', default='per_loop')
    p.add_argument('--chunk', type=int, default=8192, help='max_num_batched_tokens (the prefill chunk)')
    p.add_argument('--max-len', type=int, default=16384)
    a = p.parse_args()
    llm = LLM(model=a.model, hf_overrides={'n_recurrent_loops': a.loops, 'recurrent_kv_cache_mode': a.kv_mode},
              max_model_len=a.max_len, gpu_memory_utilization=0.3, enforce_eager=True, enable_prefix_caching=False,
              max_num_batched_tokens=a.chunk, max_num_seqs=1)
    os.makedirs(a.out_dir, exist_ok=True)
    for path in a.ids:
        ids = json.load(open(path))
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                           SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1))[0]
        pl = out.prompt_logprobs   # [None, {id: Logprob}, ...] for positions 0 .. n-1
        lp, am = [], []
        for j in range(len(ids)):
            if j + 1 < len(ids):
                d = pl[j + 1]
                lp.append(d[ids[j + 1]].logprob)
                am.append(min(d.items(), key=lambda kv: kv[1].rank)[0])
            else:
                lp.append(None)
                am.append(out.outputs[0].token_ids[0])
        name = os.path.splitext(os.path.basename(path))[0]
        json.dump({'argmax': am, 'logprob_next': lp, 'tokens': len(ids)}, open(os.path.join(a.out_dir, name + '.json'), 'w'))
        print(name, len(ids), 'tokens')


if __name__ == '__main__':
    main()
