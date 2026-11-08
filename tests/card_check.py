"""Every configs/*/card.json is well formed (XTier's tests/console_check.py for cards), no model needed.

    python tests/card_check.py

Defaults are among the choices, unavailable / note / off_when / file "when" keys and values exist, kwarg options
map every choice, files have sizes, the model directory has no path separators. Exit 1 on any problem.
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
TARGETS = {'option', 'kwarg', 'launch'}


def check(path):
    bad = []
    c = json.load(open(path, encoding='utf-8'))
    for k in ('id', 'name', 'port', 'served', 'model', 'options'):
        if k not in c:
            bad.append(f'missing {k}')
    m = c.get('model', {})
    for k in ('repo', 'dir', 'files'):
        if k not in m:
            bad.append(f'model: missing {k}')
    if '/' in m.get('dir', '') or '..' in m.get('dir', ''):
        bad.append('model.dir must be a plain name')
    for f in m.get('files', []):
        if not f.get('name') or not isinstance(f.get('size'), int) or f['size'] <= 0:
            bad.append(f'file entry {f}')
    opts = {}
    for o in c.get('options', []):
        key = o.get('key')
        values = [ch[0] for ch in o.get('choices', [])]
        opts[key] = values
        if o.get('default') not in values:
            bad.append(f'{key}: default {o.get("default")} is not a choice')
        if o.get('target', 'option') not in TARGETS:
            bad.append(f'{key}: target {o.get("target")}')
        if o.get('target') == 'kwarg':
            if not o.get('kwarg'):
                bad.append(f'{key}: kwarg option without "kwarg"')
            if set(o.get('map', {})) != set(values):
                bad.append(f'{key}: map does not cover the choices')
        for v in o.get('unavailable', {}):
            if v not in values:
                bad.append(f'{key}: unavailable {v} is not a choice')
        if o.get('default') in o.get('unavailable', {}):
            bad.append(f'{key}: the default is unavailable')
    for o in c.get('options', []):
        for cond in o.get('off_when', []):
            for k, v in cond.items():
                if k not in opts or v not in opts[k]:
                    bad.append(f'{o["key"]}: off_when {k}={v} does not exist')
    for n in c.get('notes', []):
        for k, v in n.get('when', {}).items():
            if k not in opts or v not in opts[k]:
                bad.append(f'note "{n.get("text", "")[:20]}": {k}={v} does not exist')
    for f in m.get('files', []):   # a file needed only under some option values (one context bucket's .pte)
        for k, v in f.get('when', {}).items():
            if k not in opts or v not in opts[k]:
                bad.append(f'file {f.get("name")}: when {k}={v} does not exist')
    return c.get('id'), bad


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, 'configs', '*', 'card.json')))
    ids, failed = set(), 0
    for p in paths:
        cid, bad = check(p)
        if cid in ids:
            bad.append(f'duplicate id {cid}')
        ids.add(cid)
        for b in bad:
            print('FAIL', os.path.relpath(p, ROOT), b)
        failed += bool(bad)
    print(f'{len(paths) - failed}/{len(paths)} cards ok')
    sys.exit(1 if failed or not paths else 0)


if __name__ == '__main__':
    main()
