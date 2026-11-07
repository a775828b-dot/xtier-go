"""Token ids of the check prompts: English and Chinese text and a chat prompt with tools, from the checkpoint's
tokenizer and chat template (rendered as transformers does, reference/hf_chat.py).

    python reference/make_prompts.py --model DIR --out-dir OUT [--tokens 512] [--en FILE] [--zh FILE]
"""
import argparse
import json
import os
import sys

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_chat import HfChat  # noqa: E402

TOOLS = [{'type': 'function', 'function': {
    'name': 'read_file', 'description': 'Read a file of the workspace.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Path of the file'},
                                                   'max_lines': {'type': 'integer'}}, 'required': ['path']}}}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--en')
    p.add_argument('--zh')
    a = p.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.model, 'tokenizer.json'))
    os.makedirs(a.out_dir, exist_ok=True)

    def save(name, ids):
        json.dump(ids, open(os.path.join(a.out_dir, name + '.json'), 'w'))
        print(name, len(ids))

    for name, path in (('en', a.en), ('zh', a.zh)):
        if path:
            save(name, tok.encode(open(path, encoding='utf-8').read(), add_special_tokens=False).ids[200:200 + a.tokens])
    msgs = [{'role': 'system', 'content': 'You are a careful coding assistant.'},
            {'role': 'user', 'content': 'Open src/main.cpp and tell me what the program prints.'}]
    chat = HfChat(a.model).render(msgs, tools=TOOLS)
    save('chat', tok.encode(chat, add_special_tokens=False).ids)


if __name__ == '__main__':
    main()
