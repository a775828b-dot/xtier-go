"""The C++ chat template (minja) against transformers' jinja2 rendering, character for character.

    python tests/template_check.py --cli build/xtg-cli --model MODEL_DIR

Cases: plain chat, tools with typed / enum / extra-key parameters, history with reasoning (truncated and kept),
tool calls with JSON-string arguments and consecutive tool results, enable_thinking off, medium_effort, no system
message, non-ASCII text. Needs jinja2 (reference/hf_chat.py).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference'))
from hf_chat import HfChat  # noqa: E402

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': '  Read a file.  ',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': 'Path of the file'},
            'max_lines': {'type': 'integer', 'minimum': 1},
            'mode': {'type': 'string', 'enum': ['text', 'binary']},
            'opts': {'type': 'object', 'properties': {'follow': {'type': 'boolean'}}}},
            'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'list_dir', 'parameters': {'type': 'object', 'properties': {}}}},
]
SYS = {'role': 'system', 'content': 'You are a coding agent.'}
U1 = {'role': 'user', 'content': '读一下 src/main.cpp，告诉我它打印什么。'}
A_CALL = {'role': 'assistant', 'content': '', 'reasoning_content': 'I should read the file first.',
          'tool_calls': [{'id': 'c1', 'type': 'function',
                          'function': {'name': 'read_file', 'arguments': '{"path": "src/main.cpp", "max_lines": 20, "opts": {"follow": true}}'}},
                         {'id': 'c2', 'type': 'function', 'function': {'name': 'list_dir', 'arguments': '{}'}}]}
T1 = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'int main() { puts("hi"); }'}
T2 = {'role': 'tool', 'tool_call_id': 'c2', 'content': 'src/\nREADME.md'}
A_FINAL = {'role': 'assistant', 'content': 'It prints "hi".', 'reasoning_content': 'The file calls puts.'}
U2 = {'role': 'user', 'content': 'Thanks. And with -O2?'}

CASES = [
    ('plain', [SYS, U1], None, {}),
    ('no system', [U1], None, {}),
    ('tools', [SYS, U1], TOOLS, {}),
    ('tool round', [SYS, U1, A_CALL, T1, T2], TOOLS, {}),
    ('history truncated', [SYS, U1, A_CALL, T1, T2, A_FINAL, U2], TOOLS, {}),
    ('history kept', [SYS, U1, A_CALL, T1, T2, A_FINAL, U2], TOOLS, {'truncate_history_thinking': False}),
    ('no thinking', [SYS, U1], TOOLS, {'enable_thinking': False}),
    ('medium effort', [SYS, U1, A_FINAL, U2], None, {'medium_effort': True}),
    ('content parts', [SYS, {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}], None, {}),
    ('think in content', [U1, {'role': 'assistant', 'content': '<think>x</think>done'}, U2], None, {}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    hf = HfChat(a.model)
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for name, msgs, tools, kw in CASES:
            path = os.path.join(d, 'req.json')
            json.dump({'messages': msgs, 'tools': tools or [], 'chat_template_kwargs': kw}, open(path, 'w'), ensure_ascii=False)
            out = subprocess.run([a.cli, 'render', '--model', a.model, '--request', path], capture_output=True, text=True)
            ref = hf.render(msgs, tools=tools, add_generation_prompt=True, **kw)
            if out.returncode or out.stdout != ref:
                bad += 1
                k = next((i for i in range(min(len(out.stdout), len(ref))) if out.stdout[i] != ref[i]), min(len(out.stdout), len(ref)))
                print(f'FAIL {name}: {out.stderr.strip()[:200]} first difference at {k}:')
                print('   ours', repr(out.stdout[max(0, k - 60):k + 60]))
                print('   ref ', repr(ref[max(0, k - 60):k + 60]))
            else:
                print(f'ok   {name} ({len(ref)} chars)')
    print('PASS' if not bad else f'{bad} FAILED')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
