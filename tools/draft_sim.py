"""draft_sim: the engine's prompt-lookup drafts replayed on recorded token sequences, and agent traces to replay.

    python tools/draft_sim.py check --model DIR [--request REQ.json --response RESP.json --status STATUS.json]
    python tools/draft_sim.py trace LABEL --url URL --model DIR [--sessions 3k,8k,16k,28k] [--steps 10]
                                  [--temperature 1.0] [--top-p 0.95] [--thinking on|off] [--max-tokens 8192] [--out FILE]
    python tools/draft_sim.py sim TRACE.jsonl [...] [--k 4,8,16] [--ngram 3] [--whatif]

The drafter and the accept rule follow server/runtime.cpp: the draft is what followed the most recent earlier
occurrence of the last n tokens of the context (n = 3, then 2), up to k tokens; a step feeds the last token and the
draft, keeps drafts while each equals the target's sample and adds the target's own next token. A draft is kept
only when it equals the sample, so the tokens are the ones plain decoding gives, and a recorded sequence (from any
server running the same model) yields exactly the steps the engine takes for it.

check: the indexed drafter against the plain search on random sequences, and (with --request / --response /
--status) the simulation of one engine request against the spec counters /xtg/status reported for it.
trace: tools/agent_bench.py's sessions against an OpenAI completions server that returns token ids (vLLM:
return_token_ids). The prompt is rendered here with the model's chat template (transformers) and sent as ids, the
raw output (special tokens kept) is parsed for </think> and <tool_call> blocks, and the thinking goes back into the
history as DSH sends it. One JSON line per request. Needs jinja2 and tokenizers (reference/hf_chat.py renders the
template as transformers does without loading Dory's config). sim: tokens per decode forward by part (thinking / text /
tool calls), by depth and per session, for each k.
"""
import argparse, json, random, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
import agent_bench as ab  # noqa: E402  (the sessions, the sandbox and the tools)

THINK_END, CALL, CALL_END = 13, 14, 15   # </think>, <tool_call>, </tool_call> in Dory's vocabulary
PARTS = (('t', '思考'), ('c', '正文'), ('x', '工具调用'))


def lookup_ref(ctx, last, k, ngram):
    """lookup_draft of server/runtime.cpp as written: a linear search from the end."""
    n_all = len(ctx) + 1
    at = lambda i: ctx[i] if i < len(ctx) else last
    for n in range(min(ngram, n_all - 1), 1, -1):
        for s in range(n_all - n - 1, -1, -1):
            if all(at(s + j) == at(n_all - n + j) for j in range(n)):
                d = [at(p) for p in range(s + n, min(n_all, s + n + k))]
                if d:
                    return d
    return []


LONGEST = (16, 12, 8, 6, 4, 3, 2)   # what-if: the n-gram lengths tried, longest first


class Drafter:
    """The same search with an index: the most recent start of every n-gram that lies before the current suffix.
    What-ifs, not in the engine: `ns` other n-gram lengths (tried longest first), `cands` > 1 the continuations of
    that many different recent starts, verified together as a tree."""

    def __init__(self, seq, ns, cands=1):
        self.seq, self.ns, self.cands = list(seq), sorted(ns, reverse=True), cands
        self.idx = {n: {} for n in self.ns}
        self._index(0)

    def _index(self, old):   # the starts old - n .. len - n - 1: complete n-grams, none of them the suffix itself
        s = self.seq
        for n, ix in self.idx.items():
            for p in range(max(0, old - n), len(s) - n):
                if self.cands == 1:
                    ix[tuple(s[p:p + n])] = p
                else:
                    starts = ix.setdefault(tuple(s[p:p + n]), [])
                    starts.append(p)
                    del starts[:-8]

    def extend(self, toks):
        old = len(self.seq)
        self.seq += toks
        self._index(old)

    def drafts(self, k):
        s = self.seq
        for n in self.ns:
            hit = self.idx[n].get(tuple(s[-n:])) if n <= len(s) - 1 else None
            if hit is None:
                continue
            if self.cands == 1:
                return [s[hit + n:hit + n + k]]
            out = []
            for p in reversed(hit):
                d = s[p + n:p + n + k]
                if d and d not in out:
                    out.append(d)
                    if len(out) == self.cands:
                        break
            return out
        return []


def simulate(prompt, out, k, ngram=3, longest=False, cands=1):
    """[(first, rows, accepted)] per decode forward of `out` after `prompt`: `first` is the index of the first token
    that forward gives (out[0] comes from the prefill), `rows` the draft tokens it verifies."""
    if not out:
        return []
    d = Drafter(list(prompt) + [out[0]], LONGEST if longest else range(2, ngram + 1), cands)
    p, steps = 0, []
    while p < len(out) - 1:
        best = 0
        ds = d.drafts(k)
        for dr in ds:
            a = 0
            while a < len(dr) and p + 1 + a < len(out) and dr[a] == out[p + 1 + a]:
                a += 1
            best = max(best, a)
        steps.append((p + 1, sum(map(len, ds)), best))
        d.extend(out[p + 1:p + 2 + best])
        p += best + 1
    return steps


def simulate_ref(prompt, out, k, ngram=3):
    p, steps = 0, []
    while p < len(out) - 1:
        dr = lookup_ref(list(prompt) + out[:p], out[p], k, ngram)
        a = 0
        while a < len(dr) and p + 1 + a < len(out) and dr[a] == out[p + 1 + a]:
            a += 1
        steps.append((p + 1, len(dr), a))
        p += a + 1
    return steps


def parts(out, thinking):
    """One letter per output token: t thinking (through </think>), x a tool call (<tool_call> .. </tool_call>), c text."""
    r, state = [], 't' if thinking else 'c'
    for t in out:
        if state == 't':
            r.append('t')
            state = 'c' if t == THINK_END else 't'
        elif state == 'x' or t == CALL:
            r.append('x')
            state = 'c' if t == CALL_END else 'x'
        else:
            r.append('c')
    return ''.join(r)


# --- trace ---

def parse_calls(text):
    """[(name, arguments)] from <tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>."""
    import re
    types = {t['function']['name']: t['function']['parameters']['properties'] for t in ab.TOOLS}
    calls = []
    for m in re.finditer(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', text, re.S):
        name, args = m.group(1).strip(), {}
        for p in re.finditer(r'<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>', m.group(2), re.S):
            key, val = p.group(1).strip(), p.group(2)
            if types.get(name, {}).get(key, {}).get('type') == 'integer':
                try:
                    val = int(val.strip())
                except ValueError:
                    pass
            args[key] = val
        calls.append((name, args))
    return calls


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat_and_encoder(model):
    from hf_chat import HfChat
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(Path(model) / 'tokenizer.json'))
    return HfChat(model), lambda s: tk.encode(s, add_special_tokens=False).ids


def trace(args):
    chat, enc = chat_and_encoder(args.model)
    count = lambda s: len(enc(s))
    thinking = args.thinking != 'off'
    sink, lock = open(args.out or f'trace-{args.label}.jsonl', 'a'), threading.Lock()

    def render(msgs):   # HfChat turns the tool-call arguments back into mappings, as the template expects
        return enc(chat.render(msgs, tools=ab.TOOLS, add_generation_prompt=True, enable_thinking=thinking))

    def session(name):
        target, task, files = ab.SESSIONS[name]
        sb = ab.Sandbox()
        try:
            msgs, _ = ab.history(sb, task, files, target, count)
            for step in range(args.steps):
                ids = render(msgs)
                if len(ids) + args.max_tokens > args.context_limit:
                    print(f'[{args.label} {name}] context limit at step {step} ({len(ids)} tokens)', flush=True)
                    break
                body = dict(model=args.served, prompt=ids, max_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=args.top_p, skip_special_tokens=False, return_token_ids=True)
                if args.seed is not None:
                    body['seed'] = args.seed + step
                t0 = time.time()
                ch = post(args.url + '/v1/completions', body)['choices'][0]
                out, text = list(ch.get('token_ids') or []), ch['text'].replace('<|im_end|>', '')
                if thinking:
                    i = text.find('</think>')
                    reasoning, rest = (text, '') if i < 0 else (text[:i], text[i + len('</think>'):])
                else:
                    reasoning, rest = '', text
                calls = parse_calls(rest)
                rec = dict(label=args.label, session=name, step=step, depth=len(ids), prompt=ids, out=out,
                           parts=parts(out, thinking), finish=ch.get('finish_reason'), tools=[c[0] for c in calls])
                with lock:
                    sink.write(json.dumps(rec) + '\n')
                    sink.flush()
                print(f'[{args.label} {name}] step {step}: ctx {len(ids)} out {len(out)} ({time.time() - t0:.0f} s) '
                      f'{ch.get("finish_reason")} tools {rec["tools"]}', flush=True)
                if not calls:
                    break
                hc, results = [], []
                for i, (fname, fargs) in enumerate(calls):
                    cid = f'call_{step}_{i}'
                    hc.append({'id': cid, 'type': 'function',
                               'function': {'name': fname, 'arguments': json.dumps(fargs, ensure_ascii=False)}})
                    results.append({'role': 'tool', 'tool_call_id': cid, 'content': str(sb.call(fname, fargs))[:60000]})
                turn = {'role': 'assistant', 'content': rest.split('<tool_call>')[0].strip(), 'tool_calls': hc}
                if reasoning.strip():
                    turn['reasoning_content'] = reasoning
                msgs.append(turn)
                msgs += results
        except Exception as e:  # one broken session should not stop the others
            print(f'[{args.label} {name}] failed: {type(e).__name__}: {e}', flush=True)
        finally:
            sb.close()

    threads = [threading.Thread(target=session, args=(n,)) for n in args.sessions.split(',')]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# --- sim ---

def sim(args):
    recs = [json.loads(line) for f in args.traces for line in open(f) if line.strip()]
    # columns: (name, k, longest n-gram first, candidates); --whatif adds drafters the engine does not have
    cols = [(f'k = {k}', int(k), False, 1) for k in args.k.split(',')]
    if args.whatif:
        cols += [('最长匹配 k = 16', 16, True, 1), ('最长匹配 3 条 k = 16', 16, True, 3)]
    for label in dict.fromkeys(r['label'] for r in recs):
        rs = [r for r in recs if r['label'] == label and len(r['out']) > 1]
        acc, rows = {}, {}   # group -> column -> [tokens, forwards]; column -> [draft rows, verify steps]

        def add(group, col, toks, fwd):
            g = acc.setdefault(group, {}).setdefault(col, [0, 0])
            g[0] += toks
            g[1] += fwd

        share = {}
        for r in rs:
            depth = next(f'深度 {lo // 1024}–{hi // 1024 if hi < (1 << 30) else "max"}K'
                         for lo, hi in ab.BUCKETS if lo <= r['depth'] < hi)
            for c in r['parts'][1:]:
                share[c] = share.get(c, 0) + 1
            for name, k, longest, cands in cols:
                steps = simulate(r['prompt'], r['out'], k, args.ngram, longest, cands)
                n = len(r['out']) - 1
                for g in ('全部', depth, f'会话 {r["session"]}'):
                    add(g, name, n, len(steps))
                for first, _, a in steps:
                    add(dict(PARTS)[r['parts'][first]], name, a + 1, 1)
                m = rows.setdefault(name, [0, 0])
                m[0] += sum(s[1] for s in steps)
                m[1] += sum(1 for s in steps if s[1])
        total = sum(share.values())
        print(f'\n{label}：{len(rs)} 个请求，{total} 个 decode token（思考 {share.get("t", 0) / total:.0%}，'
              f'正文 {share.get("c", 0) / total:.0%}，工具调用 {share.get("x", 0) / total:.0%}）；每次前向出的 token：')
        print('| 范围 | ' + ' | '.join(c[0] for c in cols) + ' |')
        print('|---|' + '---:|' * len(cols))
        order = ['全部'] + [p for _, p in PARTS] + sorted(g for g in acc if g.startswith('深度')) + \
            sorted(g for g in acc if g.startswith('会话'))
        for g in order:
            if g in acc:
                print(f'| {g} | ' + ' | '.join(f'{acc[g][c[0]][0] / max(1, acc[g][c[0]][1]):.2f}' for c in cols) + ' |')
        print('| 每次验证的草稿行数 | ' + ' | '.join(f'{rows[c[0]][0] / max(1, rows[c[0]][1]):.1f}' for c in cols) + ' |')


# --- check ---

def check(args):
    rng = random.Random(1)
    for _ in range(400):
        vocab = rng.choice([2, 3, 5, 30])
        seq = [rng.randrange(vocab) for _ in range(rng.randrange(4, 160))]
        cut = rng.randrange(1, len(seq) - 1)
        for k in (1, 3, 8):
            assert simulate(seq[:cut], seq[cut:], k) == simulate_ref(seq[:cut], seq[cut:], k), (seq, cut, k)
    print('indexed drafter = plain search on 400 random sequences x 3 k')
    if not args.request:
        return
    chat, enc = chat_and_encoder(args.model)
    req = json.load(open(args.request))
    kw = req.get('chat_template_kwargs') or {}
    prompt = enc(chat.render(req['messages'], add_generation_prompt=True, enable_thinking=kw.get('enable_thinking', True)))
    out = enc(json.load(open(args.response))['choices'][0]['message']['content'])
    spec = json.load(open(args.status))['telemetry']['last']['spec']
    k = json.load(open(args.status))['config']['spec_k']
    steps = simulate(prompt, out, k)
    with_draft = [s for s in steps if s[1] > 0]
    print(f'engine: {spec}; simulated from the re-tokenized text ({len(prompt)} + {len(out)} tokens, k = {k}): '
          f"{{'steps': {len(with_draft)}, 'drafted': {sum(s[1] for s in steps)}, 'accepted': {sum(s[2] for s in steps)}}}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--model')
    c.add_argument('--request')
    c.add_argument('--response')
    c.add_argument('--status')
    t = sub.add_parser('trace')
    t.add_argument('label')
    t.add_argument('--url', default='http://127.0.0.1:18401')
    t.add_argument('--served', default='dory')
    t.add_argument('--model', required=True, help='the model directory (chat template and tokenizer)')
    t.add_argument('--sessions', default='3k,8k,16k,28k')
    t.add_argument('--steps', type=int, default=10)
    t.add_argument('--max-tokens', type=int, default=8192)
    t.add_argument('--context-limit', type=int, default=65536)
    t.add_argument('--temperature', type=float, default=1.0)
    t.add_argument('--top-p', type=float, default=0.95)
    t.add_argument('--seed', type=int, default=None)
    t.add_argument('--thinking', choices=['on', 'off'], default='on')
    t.add_argument('--out')
    s = sub.add_parser('sim')
    s.add_argument('traces', nargs='+')
    s.add_argument('--k', default='4,8,16')
    s.add_argument('--ngram', type=int, default=3)
    s.add_argument('--whatif', action='store_true', help='also drafters the engine does not have (longest match, 3 candidates)')
    a = p.parse_args()
    {'check': check, 'trace': trace, 'sim': sim}[a.cmd](a)


if __name__ == '__main__':
    main()
