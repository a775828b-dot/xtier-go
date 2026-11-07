"""agent_bench: simple coding-agent sessions against xtg-server (OpenAI API with tools), a small XTier agent_bench.

Each session gives the model a small coding task on a sandbox copy of this repository, a history of earlier
read_file calls that brings the prompt to about its starting depth (3K / 8K / 16K / 28K tokens), and OpenAI tools.
The model then works as an agent for up to --steps requests: its tool calls run against the sandbox and the
results go back, so every later request re-reads a cached prefix and the context keeps growing.

Tools: read_file / list_dir / grep / edit_file / write_file / check_python (py_compile only). Nothing the model
writes is executed; every path stays inside the per-session sandbox (a temp directory, removed afterwards).

Per request, from xtg-server's /xtg/status (engine-side times): prompt / cached / output tokens, prefill and decode
tok/s, and the speculative decoding counters: tokens per decode forward = (output - 1) / (output - 1 - accepted)
(the first token comes from the prefill). Other OpenAI servers work too, without those numbers.

Usage: agent_bench.py LABEL [--url http://127.0.0.1:18400] [--sessions 3k,8k,16k,28k] [--steps 8]
       [--temperature T] [--thinking on|off] [--tokenizer MODEL_DIR] [--out FILE] [--dry-run]
Default sampling and thinking are the server's (the agent sends neither). --tokenizer (the `tokenizers`
package) sizes the histories exactly; without it they are sized at 3.2 characters per token."""
import argparse, json, os, re, shutil, sys, tempfile, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = f'{time.time_ns():x}'   # starts the system prompt: no prefix-cache reuse across runs
# sandbox contents: directories of this repository and the files taken from them
TREES = [('engine', ('*.cpp', '*.h')), ('server', ('*.cpp', '*.h')), ('packs', ('*.cpp', '*.h')),
         ('tools', ('*.py', '*.sh', '*.cpp')), ('tests', ('*.py', '*.cpp')), ('configs', ('*.json',)),
         ('docs', ('*.md',))]
TEXT = ('.cpp', '.h', '.py', '.sh', '.json', '.md', '.txt')
HIDDEN = {'tools/agent_bench.py', 'tools/draft_sim.py'}   # the harness itself stays out of the sandbox

SYSTEM = (
    '你是一个在终端里工作的编码代理（类似 Claude Code / dsh）。工作目录是 XTier Go 的源码：engine（C++ 推理引擎）、'
    'server（HTTP 服务）、packs（模型包）、tools、tests、configs、docs。\n'
    '规则：先用工具阅读相关代码再动手；修改用 edit_file（old_string 必须与文件内容逐字一致且唯一）或 write_file；'
    'Python 文件改完用 check_python 做语法检查；回答用中文，简洁。上文已经读过的文件不要重复读取；读够了就动手修改，'
    '不要只阅读不修改。完成任务后给出简短总结，不再调用工具。')

TOOLS = [
    {'type': 'function', 'function': {'name': 'read_file', 'description': '读取文件内容（带行号）。',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': '相对工作目录的路径'},
            'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'list_dir', 'description': '列出目录。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'grep', 'description': '在文件或目录中按正则搜索，返回 路径:行号:内容。',
        'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}, 'path': {'type': 'string'}},
                       'required': ['pattern']}}},
    {'type': 'function', 'function': {'name': 'edit_file', 'description': '把文件中唯一出现的 old_string 替换为 new_string。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_string': {'type': 'string'},
                       'new_string': {'type': 'string'}}, 'required': ['path', 'old_string', 'new_string']}}},
    {'type': 'function', 'function': {'name': 'write_file', 'description': '写入（覆盖）文件。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                       'required': ['path', 'content']}}},
    {'type': 'function', 'function': {'name': 'check_python', 'description': '对 Python 文件做语法检查（py_compile，不执行）。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
]

# name: (starting depth in tokens, task, files read in the pre-filled history before the padding)
SESSIONS = {
    '3k': (3_000,
           '给 tools/npu_probe/analyze.py 的 errors() 增加最大绝对误差 max_abs，并在 device() 打印的表格里加一列'
           '"h2 最大误差"。文件上文已读过，直接用 edit_file 修改，改完用 check_python 检查。',
           ['tools/npu_probe/analyze.py']),
    '8k': (8_000,
           'tests/card_check.py 检查启动台卡片。给它加两条规则：每个 option 的 default 必须是它 choices 里的值；'
           'unavailable 里的键必须出现在 choices 里。违反时报出卡片 id 和 option 的 key。相关文件上文已读过，'
           '直接修改，改完用 check_python 检查。',
           ['tests/card_check.py', 'configs/dory4b-nvfp4-sm8750/card.json']),
    '16k': (16_000,
            '照 server/runtime.cpp 里的 lookup_draft 写一个等价的 Python 实现 tools/draft_sim.py：函数 '
            'lookup_draft(ctx, last, k, ngram)，以及 simulate(tokens, prompt_len, k, ngram)，按引擎的规则模拟逐步验证，'
            '返回每步接受的草稿数列表；再写 tests/draft_sim_check.py，用几个手工构造的序列检查。相关代码上文已读过，'
            '直接用 write_file 写，写完用 check_python 检查。',
            ['server/runtime.cpp', 'server/runtime.h']),
    '28k': (28_000,
            '整理 tools/npu_probe/make_models.py：给所有函数补上中文文档字符串（说明参数和返回值），不要改变任何行为。'
            '文件上文已读过，用 edit_file 逐个函数修改，完成后用 check_python 检查。',
            ['tools/npu_probe/make_models.py', 'tools/npu_probe/build_kit.sh']),
}


class Sandbox:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix='xtg_agent_bench_'))
        for sub, pats in TREES:
            src = ROOT / sub
            for pat in pats:
                for f in src.rglob(pat):
                    rel = f.relative_to(ROOT).as_posix()
                    if '__pycache__' in f.parts or not f.is_file() or f.stat().st_size > 1_000_000 or rel in HIDDEN:
                        continue
                    t = self.root / f.relative_to(ROOT)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, t)

    def path(self, p):
        q = (self.root / str(p).lstrip('/')).resolve()
        if q != self.root and self.root not in q.parents:
            raise ValueError('路径超出工作目录')
        return q

    def call(self, name, args):
        try:
            return getattr(self, 't_' + name)(**args)
        except TypeError as e:
            return f'参数错误：{e}'
        except Exception as e:  # tool errors go back to the model, as an agent harness does
            return f'错误：{type(e).__name__}: {e}'

    def t_read_file(self, path, start_line=None, end_line=None):
        lines = self.path(path).read_text(encoding='utf-8', errors='replace').splitlines()
        a = max(1, int(start_line or 1)); b = min(len(lines), int(end_line or len(lines)), a + 1999)
        return '\n'.join(f'{i:6d}\t{lines[i - 1]}' for i in range(a, b + 1)) + \
            ('' if b >= len(lines) else f'\n... (共 {len(lines)} 行，用 start_line/end_line 继续读)')

    def t_list_dir(self, path='.'):
        d = self.path(path)
        return '\n'.join(sorted((e.name + '/') if e.is_dir() else e.name for e in d.iterdir())[:300])

    def t_grep(self, pattern, path='.'):
        rx = re.compile(pattern); base = self.path(path); out = []
        files = [base] if base.is_file() else sorted(f for f in base.rglob('*') if f.is_file() and f.suffix in TEXT)
        for f in files:
            for i, line in enumerate(f.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if rx.search(line):
                    out.append(f'{f.relative_to(self.root)}:{i}:{line[:200]}')
                    if len(out) >= 200:
                        return '\n'.join(out) + '\n...（只显示前 200 条）'
        return '\n'.join(out) or '无匹配'

    def t_edit_file(self, path, old_string, new_string):
        f = self.path(path); s = f.read_text(encoding='utf-8')
        n = s.count(old_string)
        if n != 1:
            return f'错误：old_string 出现 {n} 次（必须恰好 1 次）'
        f.write_text(s.replace(old_string, new_string), encoding='utf-8')
        return f'已修改 {path}'

    def t_write_file(self, path, content):
        f = self.path(path); f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding='utf-8')
        return f'已写入 {path}（{len(content.splitlines())} 行）'

    def t_check_python(self, path):
        import py_compile
        try:
            py_compile.compile(str(self.path(path)), doraise=True, cfile=os.devnull)
            return '语法检查通过'
        except py_compile.PyCompileError as e:
            return f'语法错误：{e.msg}'

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def counter(tokenizer):
    """Tokens of a string: the model's tokenizer when given, else about 3.2 characters per token."""
    if tokenizer:
        from tokenizers import Tokenizer
        t = Path(tokenizer)
        tok = Tokenizer.from_file(str(t / 'tokenizer.json' if t.is_dir() else t))
        return lambda s: len(tok.encode(s).ids)
    return lambda s: int(len(s) / 3.2) + 1


def history(sb, task, files, target, count):
    """System + task + pre-filled read_file calls until the prompt reaches about `target` tokens."""
    msgs = [{'role': 'system', 'content': f'（测速运行 {RUN_ID}）' + SYSTEM}, {'role': 'user', 'content': task}]
    used = count(SYSTEM + task + json.dumps(TOOLS, ensure_ascii=False))
    pool = list(files) + sorted(str(p.relative_to(sb.root)) for d in ('engine', 'packs', 'server')
                                for p in (sb.root / d).rglob('*') if p.is_file())
    seen = set()
    for i, rel in enumerate(pool):
        if used >= target:
            break
        if rel in seen:
            continue
        seen.add(rel)
        text = sb.t_read_file(rel)
        n = count(text)
        if used + n > target * 1.05 and i >= len(files):
            continue
        cid = f'call_pre_{i}'
        msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': rel})}}]})
        msgs.append({'role': 'tool', 'tool_call_id': cid, 'content': text})
        used += n + 20
    return msgs, used


def http(args, path, body=None, timeout=10):
    req = urllib.request.Request(args.url + path, json.dumps(body).encode() if body is not None else None,
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def request(msgs, args):
    body = dict(model=args.model, messages=msgs, tools=TOOLS, tool_choice='auto', max_tokens=args.max_tokens)
    if args.temperature is not None:
        body['temperature'] = args.temperature
    if args.seed is not None:
        body['seed'] = args.seed
    if args.thinking:
        body['chat_template_kwargs'] = {'enable_thinking': args.thinking == 'on'}
    t0 = time.time()
    r = http(args, '/v1/chat/completions', body, timeout=7200)
    wall = time.time() - t0
    msg = r['choices'][0]['message']
    usage = r.get('usage') or {}
    try:
        last = http(args, '/xtg/status')['telemetry']['last']
    except (urllib.error.URLError, KeyError, ValueError):
        last = {}
    out, cached = usage.get('completion_tokens', 0), (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    spec = last.get('spec') or {}
    acc = spec.get('accepted', 0)
    fwd = out - 1 - acc
    row = dict(prompt_tokens=usage.get('prompt_tokens'), cached_tokens=cached, output_tokens=out,
               prefill_s=last.get('prefill_s'), decode_s=last.get('decode_s'),
               prefill_tps=round((usage.get('prompt_tokens', 0) - cached) / last['prefill_s'], 1)
               if last.get('prefill_s') else None,
               decode_tps=round(last['decode_tok_s'], 2) if last.get('decode_tok_s') else None,
               spec_steps=spec.get('steps'), drafted=spec.get('drafted'), accepted=acc if spec else None,
               tokens_per_forward=round((out - 1) / fwd, 3) if spec and fwd > 0 else None,
               finish=r['choices'][0].get('finish_reason'), wall_s=round(wall, 2),
               tool_calls=[c['function']['name'] for c in msg.get('tool_calls') or []],
               reasoning_chars=len(msg.get('reasoning_content') or ''), text_chars=len(msg.get('content') or ''))
    return row, msg


def run_session(name, args, count):
    target, task, files = SESSIONS[name]
    sb = Sandbox()
    try:
        msgs, est = history(sb, task, files, target, count)
        print(f'[{name}] pre-filled history: {len(msgs)} messages, ~{est} tokens', flush=True)
        if args.dry_run:
            return []
        rows = []
        for step in range(args.steps):
            try:
                row, msg = request(msgs, args)
            except urllib.error.HTTPError as e:
                print(f'[{name}] step {step}: HTTP {e.code} {e.read()[:200]!r}, ending the session', flush=True)
                break
            row.update(session=name, step=step)
            rows.append(row)
            print(f'[{name}] step {step}: ctx {row["prompt_tokens"]} cached {row["cached_tokens"]} out {row["output_tokens"]} '
                  f'prefill {row["prefill_tps"]} decode {row["decode_tps"]} tok/s, {row["tokens_per_forward"]} tokens/forward, '
                  f'tools {row["tool_calls"]}', flush=True)
            calls = msg.get('tool_calls') or []
            if not calls:
                break
            results, history_calls = [], []
            for i, c in enumerate(calls):
                fn, raw = c['function'], c['function'].get('arguments') or '{}'
                cid = c.get('id') or f'call_{step}_{i}'
                try:
                    a = json.loads(raw)
                    result = sb.call(fn['name'], a) if isinstance(a, dict) else '参数必须是 JSON 对象'
                except json.JSONDecodeError as e:
                    result, raw = f'参数不是合法 JSON：{e}', '{}'   # the template renders the arguments as JSON
                history_calls.append({'id': cid, 'type': 'function', 'function': {'name': fn['name'], 'arguments': raw}})
                results.append({'role': 'tool', 'tool_call_id': cid, 'content': str(result)[:60000]})
            # the thinking goes back too, as DSH sends it: the template keeps it for the turns after the last user
            # message, so the next prompt extends this one and the prefix cache covers it (without it the prompt
            # forks inside the last generation prompt, before the sliding-window snapshot, and prefills from zero)
            turn = {'role': 'assistant', 'content': msg.get('content') or '', 'tool_calls': history_calls}
            if msg.get('reasoning_content'):
                turn['reasoning_content'] = msg['reasoning_content']
            msgs.append(turn)
            msgs += results
            grown = row['prompt_tokens'] + row['output_tokens'] + sum(count(m['content']) + 64 for m in results)
            # context guard: the next request = this prompt + this answer + the tool results + its own answer
            if grown + args.max_tokens + 512 > args.context_limit:
                print(f'[{name}] context guard: next request ~{grown} tokens, stopping the session', flush=True)
                break
        return rows
    finally:
        sb.close()


BUCKETS = ((0, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 1 << 30))


def agg(rs):
    rs = [r for r in rs if r['output_tokens'] > 1 and r['decode_s']]
    if not rs:
        return None
    toks = sum(r['output_tokens'] - 1 for r in rs)
    out = dict(requests=len(rs), output_tokens=toks + len(rs),
               decode_tps=round(toks / sum(r['decode_s'] for r in rs), 2))
    pre = [r for r in rs if r['prefill_s'] and r['prompt_tokens'] - r['cached_tokens'] > 0]
    if pre:
        out['prefill_tps'] = round(sum(r['prompt_tokens'] - r['cached_tokens'] for r in pre) / sum(r['prefill_s'] for r in pre), 1)
    if all(r['accepted'] is not None for r in rs):
        acc = sum(r['accepted'] for r in rs)
        out['tokens_per_forward'] = round(toks / max(1, toks - acc), 3)
        drafted = sum(r['drafted'] or 0 for r in rs)
        out['draft_accept_rate'] = round(acc / drafted, 3) if drafted else None
    return out


def summarize(rows):
    s = dict(all=agg(rows), depth={}, sessions={})
    for lo, hi in BUCKETS:
        a = agg([r for r in rows if lo <= (r['prompt_tokens'] or 0) < hi])
        if a:
            s['depth'][f'{lo // 1024}k-{hi // 1024 if hi < (1 << 30) else "max"}k'] = a
    for name in dict.fromkeys(r['session'] for r in rows):
        rs = [r for r in rows if r['session'] == name]
        a = agg(rs)
        if a:
            a.update(ctx_start=rs[0]['prompt_tokens'], ctx_end=rs[-1]['prompt_tokens'], steps=len(rs))
            s['sessions'][name] = a
    return s


def table(s):
    """The summary as a markdown table: one row per depth bucket and session."""
    lines = ['| 范围 | 请求 | 生成 token | prefill tok/s | decode tok/s | 每次前向出的 token | 草稿接受率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    rows = [('全部', s['all'])] + [(f'深度 {k}', v) for k, v in s['depth'].items()] + \
        [(f'会话 {k}（{v["ctx_start"]} → {v["ctx_end"]}）', v) for k, v in s['sessions'].items()]
    for name, a in rows:
        if not a:
            continue
        lines.append(f"| {name} | {a['requests']} | {a['output_tokens']} | {a.get('prefill_tps', '—')} | {a['decode_tps']} | "
                     f"{a.get('tokens_per_forward', '—')} | {a.get('draft_accept_rate') or '—'} |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('label')
    p.add_argument('--url', default=os.environ.get('AGENT_BENCH_URL', 'http://127.0.0.1:18400'))
    p.add_argument('--model', default='dory', help='served model name')
    p.add_argument('--tokenizer', default=None, help='model directory or its tokenizer.json (needs `tokenizers`)')
    p.add_argument('--out', default=None, help='result JSON (default: agent-bench-LABEL.json in the current directory)')
    p.add_argument('--sessions', default='3k,8k,16k,28k')
    p.add_argument('--context-limit', type=int, default=49152, help='server context (tokens)')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--thinking', choices=['on', 'off'], default=None)
    p.add_argument('--dry-run', action='store_true', help='build the sandboxes and histories only, no requests')
    args = p.parse_args()
    count = counter(args.tokenizer)
    rows = []; t0 = time.time()
    for name in args.sessions.split(','):
        rows += run_session(name, args, count)
    if args.dry_run:
        return
    s = summarize(rows)
    print(table(s))
    out = Path(args.out or f'agent-bench-{args.label}.json')
    json.dump(dict(label=args.label, args=vars(args), wall_s=round(time.time() - t0, 1), summary=s, rows=rows),
              open(out, 'w'), indent=1, ensure_ascii=False)
    print('saved', out)


if __name__ == '__main__':
    sys.exit(main())
