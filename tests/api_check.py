"""The HTTP server end to end, as an OpenAI client (DSH Go X) and XTier's bench scripts use it.

    python tests/api_check.py [--url http://127.0.0.1:18400] [--model dory] [--api-key KEY]

Checks: /health and /v1/models; a plain answer without thinking; a streamed tool call with reasoning (deltas,
finish_reason, usage with cached tokens); the follow-up turn with the tool result reusing the cached prefix;
/generate with prompt log probabilities; /flush_cache. Standard library only.
"""
import argparse
import json
import sys
import urllib.request

FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond:
        FAILS.append(what)


def call(url, path, body=None, key=None, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data, {'Content-Type': 'application/json'})
    if key:
        req.add_header('Authorization', 'Bearer ' + key)
    resp = urllib.request.urlopen(req, timeout=1800)
    if not stream:
        raw = resp.read().decode()
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    events = []
    for line in resp:
        line = line.decode().strip()
        if line.startswith('data:') and line != 'data: [DONE]':
            events.append(json.loads(line[5:]))
    return events


TOOLS = [{'type': 'function', 'function': {
    'name': 'read_file', 'description': 'Read a file of the workspace and return its text.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Path of the file'}},
                   'required': ['path']}}}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:18400')
    p.add_argument('--model', default='dory')
    p.add_argument('--api-key')
    a = p.parse_args()
    u, k = a.url, a.api_key

    check(urllib.request.urlopen(u + '/health', timeout=10).status == 200, '/health is 200')
    models = call(u, '/v1/models', key=k)
    check(models['data'][0]['id'] == a.model, f"/v1/models lists {a.model}")

    r = call(u, '/v1/chat/completions', dict(model=a.model, temperature=0, max_tokens=32,
                                             chat_template_kwargs={'enable_thinking': False},
                                             messages=[{'role': 'user', 'content': 'What is 2+3? Reply with the number only.'}]), k)
    text = r['choices'][0]['message']['content'] or ''
    check('5' in text, f'plain answer without thinking: {text!r}')

    msgs = [{'role': 'system', 'content': 'You are a coding agent. Use the tools.'},
            {'role': 'user', 'content': 'Read the file src/main.cpp and tell me what it prints.'}]
    ev = call(u, '/v1/chat/completions', dict(model=a.model, messages=msgs, tools=TOOLS, temperature=0, max_tokens=1024,
                                              stream=True, stream_options={'include_usage': True}), k, stream=True)
    reasoning = ''.join(c['delta'].get('reasoning_content', '') for e in ev for c in e.get('choices', []))
    content = ''.join(c['delta'].get('content', '') or '' for e in ev for c in e.get('choices', []))
    calls = {}
    for e in ev:
        for c in e.get('choices', []):
            for t in c['delta'].get('tool_calls', []):
                d = calls.setdefault(t['index'], {'id': None, 'name': None, 'args': ''})
                d['id'] = t.get('id') or d['id']
                d['name'] = (t.get('function') or {}).get('name') or d['name']
                d['args'] += (t.get('function') or {}).get('arguments', '')
    finish = [c['finish_reason'] for e in ev for c in e.get('choices', []) if c.get('finish_reason')]
    usage = [e['usage'] for e in ev if e.get('usage')]
    check(len(reasoning) > 0, f'streamed reasoning ({len(reasoning)} chars)')
    check(len(calls) == 1 and calls[0]['name'] == 'read_file', f'one read_file call: {calls}')
    args = json.loads(calls[0]['args']) if calls else {}
    check(args.get('path') == 'src/main.cpp', f'call arguments {args}')
    check(finish == ['tool_calls'], f'finish_reason {finish}')
    check(len(usage) == 1 and usage[0]['completion_tokens'] > 0, f'usage chunk {usage}')

    msgs += [{'role': 'assistant', 'content': content, 'reasoning_content': reasoning,
              'tool_calls': [{'id': calls[0]['id'], 'type': 'function',
                              'function': {'name': 'read_file', 'arguments': calls[0]['args']}}]},
             {'role': 'tool', 'tool_call_id': calls[0]['id'],
              'content': '#include <cstdio>\nint main() { std::puts("hello from xtier-go"); }\n'}]
    r = call(u, '/v1/chat/completions', dict(model=a.model, messages=msgs, tools=TOOLS, temperature=0, max_tokens=1024), k)
    m = r['choices'][0]['message']
    cached = r['usage']['prompt_tokens_details']['cached_tokens']
    check('hello from xtier-go' in (m.get('content') or ''), f"answer uses the tool result: {(m.get('content') or '')[:120]!r}")
    check(cached > usage[0]['prompt_tokens'] * 0.9 if usage else False,
          f"follow-up reuses the cached prefix: {cached} of {r['usage']['prompt_tokens']} prompt tokens")

    ids = call(u, '/generate', dict(text='The capital of France is', sampling_params=dict(max_new_tokens=4, temperature=0),
                                   return_logprob=True, logprob_start_len=0), k)
    lp = ids['meta_info']['input_token_logprobs']
    check(lp[0][0] is None and all(x[0] is not None for x in lp[1:]) and len(lp) == ids['meta_info']['prompt_tokens'],
          f"/generate input logprobs ({len(lp)} entries), text {ids['text']!r}")
    check(call(u, '/flush_cache', {}, k).startswith('Cache flushed'), '/flush_cache')
    print('PASS' if not FAILS else f'{len(FAILS)} FAILED')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
