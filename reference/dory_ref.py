"""Dory in plain PyTorch, written from vLLM's dory.py, for checking xtier-go (not for serving).

    python reference/dory_ref.py --model DIR --ids IDS.json [--loops 2] [--kv-mode per_loop] [--chunk N]
                                 [--bf16] [--device cuda] --out OUT.json [--logits OUT.f32]

NVFP4 weights are decoded to FP32 at load (E2M1 code, low nibble first, times the E4M3 block scale times the
tensor scale). The forward goes chunk by chunk with explicit per-instance KV caches, so last_loop and chunked
prefill behave as in vLLM (and as in the engine). --bf16 rounds where vLLM's BF16 model does (activations
between operations, the RoPE cos / sin table): compare with vLLM in that mode, with the engine in FP32.
Output: per position the log probability of the next token and the argmax (the same JSON as xtg-cli forward).
"""
import argparse
import json
import math
import os

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def dequant(f, name, device):
    w = f.get_tensor(name + '.weight')                       # uint8 [n, k / 2]
    s = f.get_tensor(name + '.weight_scale').float()          # e4m3 [n, k / 16]
    g = f.get_tensor(name + '.weight_scale_2').float()        # scalar
    n, half = w.shape
    codes = torch.stack([w & 15, w >> 4], dim=-1).reshape(n, half * 2).long()
    return (E2M1[codes] * s.repeat_interleave(16, dim=1) * g).to(device)


class Ref:
    def __init__(self, model_dir, loops=None, kv_mode=None, bf16=False, device='cpu', dump=None):
        c = json.load(open(os.path.join(model_dir, 'config.json')))
        self.c = c
        self.dump = dump
        self.calls = {}
        self.dev = device
        self.dt = torch.bfloat16 if bf16 else torch.float32
        self.loops = loops or c['n_recurrent_loops']
        self.per_loop = (kv_mode or c['recurrent_kv_cache_mode']) == 'per_loop'
        self.H, self.F = c['hidden_size'], c['intermediate_size']
        self.nh, self.nkv, self.hd = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        self.n_in, self.n_rec = c['n_input_layers'], c['n_recurrent_layers']
        self.window = c['sliding_window']
        self.types = c['layer_types']
        std = c['initializer_range']
        self.alpha_scale = c['ngpt_alpha_init'] / std
        g0 = c.get('attention_output_gate_logit_scale_init') or c['hidden_size'] ** 0.5
        sqk_factor, gate_factor, suv_factor = self.hd ** 0.5 / std, g0 ** 0.5 / std, self.H ** 0.5
        f = safe_open(os.path.join(model_dir, 'model.safetensors'), 'pt')
        t = lambda n: f.get_tensor(n).to(device)
        self.embed = t('backbone.embeddings.weight').to(self.dt)
        self.lm_head = t('lm_head.weight').to(self.dt)
        self.logit_scale = t('lm_head.logit_scale').float() / std
        self.layers = []
        for i, kind in enumerate(self.types):
            p = f'backbone.layers.{i}.'
            L = {'kind': kind}
            if kind == 'mlp':
                for w in ('gate_proj', 'up_proj', 'down_proj'):
                    L[w] = dequant(f, p + 'mixer.' + w, device).to(self.dt)
                L['sg'] = t(p + 'mixer.suv_gate').float() * suv_factor
                L['su'] = t(p + 'mixer.suv_up').float() * suv_factor
                L['alpha'] = t(p + 'mlp_alpha')
            else:
                for w in ('q_proj', 'k_proj', 'v_proj', 'g_proj', 'o_proj'):
                    L[w] = dequant(f, p + 'mixer.' + w, device).to(self.dt)
                L['qs'] = (t(p + 'mixer.sqk').float() * sqk_factor).square()
                L['gs'] = (t(p + 'mixer.s_gate').float() * gate_factor).square()
                L['alpha'] = t(p + 'attn_alpha')
                theta = c['rope_parameters'][kind]['rope_theta']
                L['inv_freq'] = (1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float) / self.hd))).to(device)
            self.layers.append(L)
        self.reset()

    def reset(self):
        self.n = 0
        self.cache = {}   # (layer, instance) -> [K, V] over all positions (sliding layers read only their window)

    # vLLM's helpers
    def justnorm(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

    def slerp(self, a, b, alpha):
        ya, yb = self.justnorm(a).float(), self.justnorm(b).float()
        lr = (alpha.float() * self.alpha_scale).abs()
        return self.justnorm(ya + lr * (yb - ya)).to(a.dtype)

    def rope(self, x, pos, inv_freq):
        # x [t, heads, d]; vLLM's table is FP32 cos / sin of FP32(pos) * inv_freq, cast to the model dtype
        freqs = torch.einsum('i,j->ij', pos.float(), inv_freq)
        cos, sin = freqs.cos().to(self.dt), freqs.sin().to(self.dt)
        half = self.hd // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos, sin = cos[:, None, :], sin[:, None, :]
        if self.dt == torch.bfloat16:
            o1 = (x1.float() * cos.float() - x2.float() * sin.float()).to(self.dt)
            o2 = (x2.float() * cos.float() + x1.float() * sin.float()).to(self.dt)
        else:
            o1, o2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
        return torch.cat([o1, o2], dim=-1)

    def mm(self, x, w):
        return (x.float() @ w.float().T).to(self.dt) if self.dt == torch.bfloat16 else x @ w.T

    def attention(self, li, inst, q, k, v, pos0, store):
        t = q.shape[0]
        key = (li, inst)
        if key not in self.cache:
            self.cache[key] = [k.new_zeros(0, self.nkv, self.hd), v.new_zeros(0, self.nkv, self.hd)]
        K = torch.cat([self.cache[key][0][:pos0], k], 0)
        V = torch.cat([self.cache[key][1][:pos0], v], 0)
        if store:
            self.cache[key] = [K, V]
        n = K.shape[0]
        qpos = torch.arange(pos0, pos0 + t, device=q.device)[:, None]
        kpos = torch.arange(n, device=q.device)[None, :]
        mask = kpos <= qpos
        if self.types[li] == 'sliding_attention':
            mask &= kpos > qpos - self.window
        rep = self.nh // self.nkv
        Kr = K.float().repeat_interleave(rep, dim=1)       # [n, nh, d]
        Vr = V.float().repeat_interleave(rep, dim=1)
        s = torch.einsum('thd,nhd->htn', q.float(), Kr) / math.sqrt(self.hd)
        s = s.masked_fill(~mask[None], float('-inf'))
        p = torch.softmax(s, dim=-1)
        return torch.einsum('htn,nhd->thd', p, Vr).to(self.dt)

    def sublayer(self, li, x, pos, inst, store):
        out = self._sublayer(li, x, pos, inst, store)
        if self.dump and self.n == 0:
            k = self.calls.get(li, 0)
            self.calls[li] = k + 1
            torch.save(out.float().cpu(), os.path.join(self.dump, f'layer{li}_call{k}.pt'))
        return out

    def _sublayer(self, li, x, pos, inst, store):
        L = self.layers[li]
        t = x.shape[0]
        if L['kind'] == 'mlp':
            g, u = self.mm(x, L['gate_proj']), self.mm(x, L['up_proj'])
            g = (g.float() * L['sg']).to(self.dt)
            u = (u.float() * L['su']).to(self.dt)
            m = (torch.nn.functional.silu(g.float()) * u.float()).to(self.dt)
            m = self.justnorm(m)
            return self.slerp(x, self.mm(m, L['down_proj']), L['alpha'])
        q = self.mm(x, L['q_proj']).view(t, self.nh, self.hd)
        k = self.mm(x, L['k_proj']).view(t, self.nkv, self.hd)
        v = self.mm(x, L['v_proj']).view(t, self.nkv, self.hd)
        gate = self.mm(x, L['g_proj'])
        q, k = self.rope(q, pos, L['inv_freq']), self.rope(k, pos, L['inv_freq'])
        q = (L['qs'].view(1, self.nh, self.hd) * self.justnorm(q).float()).to(self.dt)
        k = self.justnorm(k)
        out = self.attention(li, inst, q, k, v, int(pos[0]), store).reshape(t, -1)
        out = (out.float() * torch.sigmoid(gate.float() * L['gs'])).to(self.dt)
        out = self.justnorm(out)
        return self.slerp(x, self.mm(out, L['o_proj']), L['alpha'])

    @torch.no_grad()
    def forward(self, ids):
        """ids of the next positions -> logits [t, vocab] (FP32)."""
        t = len(ids)
        pos = torch.arange(self.n, self.n + t, device=self.dev)
        h = self.embed[torch.tensor(ids, device=self.dev)]
        for li in range(self.n_in):
            h = self.sublayer(li, h, pos, 0, True)
        input_h = h
        state = input_h
        for i in range(self.loops):
            h = self.justnorm(state + input_h)
            store = self.per_loop or i == self.loops - 1
            for li in range(self.n_in, self.n_in + self.n_rec):
                h = self.sublayer(li, h, pos, i if self.per_loop else 0, store)
            state = self.justnorm(h)
        h = state
        for li in range(self.n_in + self.n_rec, len(self.types)):
            h = self.sublayer(li, h, pos, 0, True)
        self.n += t
        logits = self.mm(h, self.lm_head) if self.dt == torch.bfloat16 else h @ self.lm_head.T
        return logits.float() * self.logit_scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--loops', type=int)
    p.add_argument('--kv-mode')
    p.add_argument('--chunk', type=int, default=0, help='prefill chunk (0: the whole prompt at once)')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', required=True)
    p.add_argument('--logits')
    p.add_argument('--dump', help='save every sublayer output of the first chunk here')
    a = p.parse_args()
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
    torch.set_num_threads(min(64, os.cpu_count()))
    ids = json.load(open(a.ids))
    m = Ref(a.model, a.loops, a.kv_mode, a.bf16, a.device, a.dump)
    chunk = a.chunk or len(ids)
    lp, am = [], []
    sink = open(a.logits, 'wb') if a.logits else None
    for s in range(0, len(ids), chunk):
        lg = m.forward(ids[s:s + chunk])
        logp = torch.log_softmax(lg, -1)
        for r in range(lg.shape[0]):
            j = s + r
            am.append(int(lg[r].argmax()))
            lp.append(float(logp[r, ids[j + 1]]) if j + 1 < len(ids) else None)
        if sink:
            sink.write(lg.cpu().numpy().astype('float32').tobytes())
    json.dump({'argmax': am, 'logprob_next': lp, 'tokens': len(ids)}, open(a.out, 'w'))
    print(a.out, len(ids), 'tokens')


if __name__ == '__main__':
    main()
