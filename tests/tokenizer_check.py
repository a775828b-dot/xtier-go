"""The C++ tokenizer against Hugging Face tokenizers, token for token.

    python tests/tokenizer_check.py --cli build/xtg-cli --model MODEL_DIR [--files F ...]

Texts: built-in edge cases (whitespace runs, CR/LF, numbers, punctuation, special tokens written in the text,
several scripts with combining marks, emoji) plus the given files cut into pieces. Needs `tokenizers`.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

CASES = [
    '', ' ', '  ', '\n', '\n\n', ' \n', '\r\n', 'a\r\n\r\nb', '\t\tx', 'x  ', 'x \n y', '  \n  x',
    'Hello world', ' Hello', 'HELLO world', 'HTTPServer', "don't I'LL can't", 'snake_case_name camelCaseName',
    '12345 3.14159 1,000,000 0x7f', '¹²³ ½', '你好，世界！这是一个测试。', '日本語のテキストとカタカナ', '한국어 텍스트',
    'नमस्ते दुनिया', 'مرحبا بالعالم', 'Ελληνικά κείμενο', 'émigré façade naïve', 'e\u0301 combining', '🙂👍🏽 family 👨‍👩‍👧',
    '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n', '<tool_call>\n<function=read_file>\n'
    '<parameter=path>\nsrc/a.cpp\n</parameter>\n</function>\n</tool_call>', '<s></s>[INST]x[/INST]<SPECIAL_999>',
    'def f(x):\n    return x ** 2  # comment\n\n\nclass A:\n\tpass\n', '    \n\n    \t', 'a/b/c\n//', '...!!!???',
    '\u3000全角\u3000空格', 'mixed中文English混合', '\u00a0nbsp\u2009thin',
]


def pieces(text, n=40):
    step = max(1, len(text) // n)
    return [text[i:i + step * 2] for i in range(0, len(text), step)][:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--files', nargs='*', default=[])
    a = p.parse_args()
    hf = Tokenizer.from_file(os.path.join(a.model, 'tokenizer.json'))
    texts = list(CASES)
    for f in a.files:
        t = open(f, encoding='utf-8', errors='ignore').read()
        texts.append(t[:200000])
        texts += pieces(t)
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(texts):
            path = os.path.join(d, 'in.txt')
            with open(path, 'w', encoding='utf-8', newline='') as o:
                o.write(t)
            out = subprocess.run([a.cli, 'tokenize', '--tokenizer', os.path.join(a.model, 'tokenizer.json'), '--in', path],
                                 capture_output=True, text=True)
            if out.returncode:
                print('FAIL', i, out.stderr.strip())
                bad += 1
                continue
            ours = json.loads(out.stdout)
            ref = hf.encode(t, add_special_tokens=False).ids
            if ours != ref:
                bad += 1
                k = next((j for j in range(min(len(ours), len(ref))) if ours[j] != ref[j]), min(len(ours), len(ref)))
                print(f'FAIL {i}: {len(ours)} vs {len(ref)} tokens, first difference at {k}: {repr(t[:60])}')
                print('   ours', ours[max(0, k - 3):k + 5], [hf.id_to_token(x) for x in ours[max(0, k - 3):k + 5]])
                print('   ref ', ref[max(0, k - 3):k + 5], [hf.id_to_token(x) for x in ref[max(0, k - 3):k + 5]])
    print(f'{len(texts) - bad}/{len(texts)} texts identical')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
