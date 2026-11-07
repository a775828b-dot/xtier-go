"""Every sublayer's output inside vLLM for one prompt, to find where an implementation departs from it.

    python reference/vllm_layers.py --model DIR --ids IDS.json --dump OUT_DIR [--loops 2] [--kv-mode per_loop]

Hooks DoryDecoderLayer in the worker and saves layer<i>_call<k>.pt (FP32 [tokens, hidden]) for the prefill;
reference/dory_ref.py --dump writes the same files, tests/layers_check.py compares the two directories.
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--dump', required=True)
    p.add_argument('--loops', type=int, default=2)
    p.add_argument('--kv-mode', default='per_loop')
    a = p.parse_args()
    os.makedirs(a.dump, exist_ok=True)
    llm = LLM(model=a.model, hf_overrides={'n_recurrent_loops': a.loops, 'recurrent_kv_cache_mode': a.kv_mode},
              max_model_len=8192, gpu_memory_utilization=0.3, enforce_eager=True, enable_prefix_caching=False,
              max_num_seqs=1)
    ids = json.load(open(a.ids))
    dump, n = a.dump, len(ids)

    def install(model):
        import torch
        calls = {}

        def make(i):
            def hook(mod, args, out):
                if out.shape[0] != n:
                    return
                k = calls.get(i, 0)
                calls[i] = k + 1
                torch.save(out.detach().float().cpu(), os.path.join(dump, f'layer{i}_call{k}.pt'))
            return hook

        for i, layer in enumerate(model.backbone.layers):
            layer.register_forward_hook(make(i))

        def emb_hook(mod, args, out):
            if out.shape[0] == n:
                torch.save(out.detach().float().cpu(), os.path.join(dump, 'embed.pt'))
        model.backbone.embeddings.register_forward_hook(emb_hook)
        return True

    llm.apply_model(install)
    llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0))
    print(len(os.listdir(dump)), 'tensors in', dump)


if __name__ == '__main__':
    main()
