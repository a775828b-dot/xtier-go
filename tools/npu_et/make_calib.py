"""make_calib.py OUT.json [N] [PROMPT_TOKENS] [OUT_TOKENS]: calibration conversations for ExecuTorch's static LLM
export (--calib_samples), decoded from the S1 agent traces (Dory token ids) so the activations seen in calibration
look like agent work: tool output, code, reasoning. One sample per trace record, sessions taken in turn."""
import json
import os
import sys

from tokenizers import Tokenizer

out_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
p_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
o_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 800
tok = Tokenizer.from_file(os.environ['DORY_TOKENIZER'])
recs = []
for f in ('s1-think', 's1-nothink', 's1-greedy'):
    recs += [json.loads(l) for l in open(os.path.join(os.environ['DORY_TRACES'], f + '.jsonl'))]
by_session = {}
for r in recs:
    by_session.setdefault(r['session'], []).append(r)
order = []
while len(order) < n and any(by_session.values()):
    for s in sorted(by_session):
        if by_session[s] and len(order) < n:
            order.append(by_session[s].pop(0))
samples = []
for r in order:
    user = tok.decode(r['prompt'][-p_tok:], skip_special_tokens=True)
    reply = tok.decode(r['out'][:o_tok], skip_special_tokens=True)
    samples.append({'messages': [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]})
json.dump(samples, open(out_path, 'w'), ensure_ascii=False, indent=1)
print(out_path, len(samples), 'samples', [r['session'] for r in order])
