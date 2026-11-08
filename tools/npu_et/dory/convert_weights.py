# Dory BF16 checkpoint (Hugging Face) -> DoryModel state dict, with the nGPT scales folded (see model.py).
#
#   python -m executorch.examples.qualcomm.oss_scripts.dory.convert_weights MODEL_DIR OUT.pth
#
# The BF16 release (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) is the source: the NVFP4 one would be quantized
# twice on the way to 4-bit integer blocks.

import json
import math
import os
import sys

import torch
from safetensors import safe_open


def convert_weights(input_dir: str, output_file: str) -> None:
    torch.save(load_state_dict(input_dir), output_file)


def load_state_dict(input_dir: str) -> dict:
    c = json.load(open(os.path.join(input_dir, "config.json")))
    H, F, hd, nh = c["hidden_size"], c["intermediate_size"], c["head_dim"], c["num_attention_heads"]
    std = c["initializer_range"]
    alpha_scale = c["ngpt_alpha_init"] / std
    g0 = c.get("attention_output_gate_logit_scale_init") or H**0.5
    sqk_factor, gate_factor, suv_factor = hd**0.5 / std, g0**0.5 / std, H**0.5
    f = safe_open(os.path.join(input_dir, "model.safetensors"), "pt")
    t = lambda n: f.get_tensor(n).float()
    conv = lambda w: w.reshape(*w.shape, 1, 1).contiguous()
    unit = lambda d: torch.full((d,), 1 / math.sqrt(d))
    sd = {"tok_embeddings.weight": t("backbone.embeddings.weight"), "loop_norm.weight": unit(H)}
    sd["output.weight"] = t("lm_head.weight") * (t("lm_head.logit_scale") / std)[:, None]
    # one entry per evaluation (model.py unrolls the loops, each with its own copy of the recurrent weights)
    n_in, n_rec = c["n_input_layers"], c["n_recurrent_layers"]
    phys = (list(range(n_in)) + list(range(n_in, n_in + n_rec)) * c["n_recurrent_loops"] +
            list(range(n_in + n_rec, len(c["layer_types"]))))
    for i, layer in enumerate(phys):
        kind = c["layer_types"][layer]
        p, o = f"backbone.layers.{layer}.", f"sublayers.{i}."
        if kind == "mlp":
            sd[o + "wg.weight"] = conv(t(p + "mixer.gate_proj.weight") * (t(p + "mixer.suv_gate") * suv_factor)[:, None])
            sd[o + "wu.weight"] = conv(t(p + "mixer.up_proj.weight") * (t(p + "mixer.suv_up") * suv_factor)[:, None])
            sd[o + "wd.weight"] = conv(t(p + "mixer.down_proj.weight"))
            sd[o + "m_norm.weight"] = unit(F)
            alpha = t(p + "mlp_alpha")
        else:
            for w in ("q", "k", "v", "o"):
                sd[o + f"w{w}.weight"] = conv(t(p + f"mixer.{w}_proj.weight"))
            gs = (t(p + "mixer.s_gate") * gate_factor).square()
            sd[o + "wg.weight"] = conv(t(p + "mixer.g_proj.weight") * gs[:, None])
            qs = (t(p + "mixer.sqk") * sqk_factor).square() / math.sqrt(hd)   # the attention scale too
            sd[o + "qs"] = qs.view(1, nh, 1, hd)
            sd[o + "q_norm.weight"] = unit(hd)
            sd[o + "k_norm.weight"] = unit(hd)
            sd[o + "o_norm.weight"] = unit(nh * hd)
            alpha = t(p + "attn_alpha")
        lr = (alpha * alpha_scale).abs()
        sd[o + "res.keep"] = 1 - lr
        sd[o + "res.b_norm.weight"] = lr / math.sqrt(H)
        sd[o + "res.out_norm.weight"] = unit(H)
        if i == 0:   # the embedding row is not exactly unit
            sd[o + "res.x_norm.weight"] = unit(H)
    return sd


if __name__ == "__main__":
    convert_weights(sys.argv[1], sys.argv[2])
