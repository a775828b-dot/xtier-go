"""The chat template rendered as transformers renders it (apply_chat_template), without loading the model config
(transformers does not know Dory's): the same jinja2 environment, tojson filter and special-token variables.
vLLM's preprocessing (tool call arguments from JSON strings to objects) is applied first.
"""
import json
import os

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


class HfChat:
    def __init__(self, model_dir):
        path = os.path.join(model_dir, 'chat_template.jinja')
        cfg = json.load(open(os.path.join(model_dir, 'tokenizer_config.json')))
        self.source = open(path).read() if os.path.exists(path) else cfg['chat_template']
        tok = lambda v: v if isinstance(v, str) else (v or {}).get('content')
        self.special = {k: tok(cfg.get(k)) for k in ('bos_token', 'eos_token', 'unk_token', 'pad_token') if cfg.get(k)}
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters['tojson'] = _tojson
        env.globals['raise_exception'] = _raise
        self.template = env.from_string(self.source)

    @staticmethod
    def prepare(messages):
        out = []
        for m in messages:
            m = json.loads(json.dumps(m))
            if m.get('role') == 'assistant' and isinstance(m.get('tool_calls'), list):
                for c in m['tool_calls']:
                    f = c.get('function', {})
                    if isinstance(f.get('arguments'), str):
                        f['arguments'] = json.loads(f['arguments'])
            if isinstance(m.get('content'), list):
                m['content'] = '\n'.join(p.get('text', '') for p in m['content'] if p.get('type') == 'text')
            out.append(m)
        return out

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        return self.template.render(messages=self.prepare(messages), tools=tools,
                                    add_generation_prompt=add_generation_prompt, **self.special, **kwargs)
