"""apply_dory_patch.py EXECUTORCH_ROOT: register Dory ("dory-4b") in the Qualcomm static LLM example.

Edits (each checked against its anchor, skipped when already there):
  llama/decoder_constants.py      DECODER_MODEL_VERSION "dory-4b" -> "dory"
  llama/static_llm_quant_recipe.py DoryQuantRecipe: 16a4w, convs 16a4w blocks of 16, lm_head 16a8w per channel, 8-bit KV
  llama/__init__.py               the LLMModelConfig and LLM_VARIANT_ARCHS entry
  llama/wrappers/llm_wrappers.py  DoryConfig (Hugging Face config.json) in place of ModelArgs
  llama/runner/runner.cpp, runner/utils.h?  "dory": hybrid cache (two masks), EOS <|im_end|> and </s>
  llama/qnn_llama_runner.cpp      the ChatML prompt of Dory (thinking on: "<think>\n" after the assistant tag)
"""
import os
import re
import sys

root = sys.argv[1]
L = os.path.join(root, "examples/qualcomm/oss_scripts/llama")


def edit(rel, anchor, new, once_marker):
    p = os.path.join(L, rel)
    s = open(p).read()
    if once_marker in s:
        print("already", rel, once_marker[:40])
        return
    assert s.count(anchor) == 1, f"{rel}: anchor found {s.count(anchor)} times: {anchor[:60]!r}"
    s = s.replace(anchor, new)
    open(p, "w").write(s)
    print("patched", rel)


edit("decoder_constants.py", '    "gemma4-e2b": "gemma4",\n',
     '    "gemma4-e2b": "gemma4",\n    "dory-4b": "dory",\n', '"dory-4b"')

recipe = '''

class DoryQuantRecipe(StaticLLMQuantRecipe):
    """Dory 4B: 16-bit activations, 4-bit weights in blocks of 16 along the input (Dory was trained with NVFP4
    blocks of 16), the lm_head 16a8w per channel, 8-bit KV."""

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {
                    r"output\\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)
'''
p = os.path.join(L, "static_llm_quant_recipe.py")
s = open(p).read()
if "class DoryQuantRecipe" not in s:
    open(p, "w").write(s.rstrip("\n") + "\n" + recipe)
    print("patched static_llm_quant_recipe.py")
else:
    print("already static_llm_quant_recipe.py")

edit("__init__.py", "    CodegenQuantRecipe,\n", "    CodegenQuantRecipe,\n    DoryQuantRecipe,\n", "DoryQuantRecipe,")
edit("__init__.py",
     "from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (\n",
     "from executorch.examples.qualcomm.oss_scripts.dory.convert_weights import (\n"
     "    convert_weights as convert_dory_weights,\n)\n"
     "from executorch.examples.qualcomm.oss_scripts.dory.model import DoryModel\n"
     "from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (\n",
     "convert_dory_weights")
edit("__init__.py", '    "gemma2-2b": MultiScopeAwareLlamaModel,\n}',
     '    "gemma2-2b": MultiScopeAwareLlamaModel,\n    "dory-4b": DoryModel,\n}', '"dory-4b": DoryModel')
p = os.path.join(L, "__init__.py")
s = open(p).read()
if '@register_llm_model("dory-4b")' not in s:
    s = s.rstrip("\n") + '''


@register_llm_model("dory-4b")
@dataclass(init=False, frozen=True)
class Dory_4B(LLMModelConfig):
    repo_id: str = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
    params_path: str = os.path.join(BASE_DIR, "../dory/config/4b_config.json")
    convert_weights = convert_dory_weights
    transform_weight = False
    instruct_model = True
    num_sharding = 1
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quant_recipe = DoryQuantRecipe
'''
    open(p, "w").write(s)
    print("patched __init__.py (config)")
else:
    print("already __init__.py (config)")

edit("wrappers/llm_wrappers.py",
     '        else:\n            params_path = (\n                config.params_path\n',
     '        elif control_args.decoder_model == "dory-4b":\n'
     '            from executorch.examples.qualcomm.oss_scripts.dory.model import DoryConfig\n\n'
     '            params_path = (\n                config.params_path\n                if control_args.params is None\n'
     '                else control_args.params\n            )\n'
     '            self.model_args = process_model_args(\n                control_args,\n'
     '                DoryConfig.from_json(params_path),\n                self.quant_recipe,\n                config,\n'
     '                mode,\n            )\n'
     '        else:\n            params_path = (\n                config.params_path\n',
     "DoryConfig.from_json(params_path)")

# runner: the decoder version, hybrid cache, EOS
edit("runner/runner.cpp",
     '  } else if (decoder_model_version == "granite") {\n',
     '  } else if (decoder_model_version == "dory") {\n'
     '    decoder_model_version_ = DecoderModelVersion::kDory;\n'
     '    cache_mode_ = CacheMode::HybridCache;\n'
     '  } else if (decoder_model_version == "granite") {\n',
     "DecoderModelVersion::kDory;")
edit("runner/runner.cpp",
     '  } else if (decoder_model_version_ == DecoderModelVersion::kGemma4) {\n',
     '  } else if (decoder_model_version_ == DecoderModelVersion::kDory) {\n'
     '    eos_ids->insert(tokenizer_->encode("<|im_end|>", 0, 0).get()[0]);\n'
     '    eos_ids->insert(tokenizer_->encode("</s>", 0, 0).get()[0]);\n'
     '  } else if (decoder_model_version_ == DecoderModelVersion::kGemma4) {\n',
     "DecoderModelVersion::kDory) {")
hdr = None
for f in os.listdir(os.path.join(L, "runner")):
    if f.endswith(".h") and "kGemma4" in open(os.path.join(L, "runner", f)).read():
        hdr = f
assert hdr, "no header declares kGemma4"
p = os.path.join(L, "runner", hdr)
s = open(p).read()
if "kDory" not in s:
    s2 = re.sub(r"(\n(\s*)kGemma4,)", r"\1\n\2kDory,", s, count=1)
    assert s2 != s, f"{hdr}: kGemma4, not found"
    open(p, "w").write(s2)
    print("patched runner/" + hdr)
else:
    print("already runner/" + hdr)
edit("qnn_llama_runner.cpp",
     "    case example::DecoderModelVersion::kGemma4:\n",
     "    case example::DecoderModelVersion::kDory:\n"
     "      // Dory's template always has a system turn, empty without a system prompt\n"
     '      formatted_prompt.append("<|im_start|>system\\n");\n'
     "      formatted_prompt.append(system_prompt);\n"
     '      formatted_prompt.append("<|im_end|>\\n");\n'
     '      formatted_prompt.append("<|im_start|>user\\n");\n'
     "      formatted_prompt.append(prompt);\n"
     '      formatted_prompt.append("<|im_end|>\\n<|im_start|>assistant\\n<think>\\n");\n'
     "      break;\n"
     "    case example::DecoderModelVersion::kGemma4:\n",
     "DecoderModelVersion::kDory:")
# "dory-tiny": the first 8 sublayers (2 input, 4 looped, 2 output) for quick pipeline tests, from the offline
# Hugging Face repo local/dory-tiny (tiny config.json, the real weights linked)
edit("decoder_constants.py", '    "dory-4b": "dory",\n', '    "dory-4b": "dory",\n    "dory-tiny": "dory",\n', '"dory-tiny"')
edit("__init__.py", '    "dory-4b": DoryModel,\n', '    "dory-4b": DoryModel,\n    "dory-tiny": DoryModel,\n',
     '"dory-tiny": DoryModel')
p = os.path.join(L, "__init__.py")
s = open(p).read()
if '@register_llm_model("dory-tiny")' not in s:
    s = s.rstrip("\n") + '''


@register_llm_model("dory-tiny")
@dataclass(init=False, frozen=True)
class Dory_Tiny(LLMModelConfig):
    repo_id: str = "local/dory-tiny"
    params_path: str = os.path.join(BASE_DIR, "../dory/config/tiny_config.json")
    convert_weights = convert_dory_weights
    transform_weight = False
    instruct_model = True
    num_sharding = 1
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quant_recipe = DoryQuantRecipe
'''
    open(p, "w").write(s)
    print("patched __init__.py (tiny config)")
else:
    print("already __init__.py (tiny config)")
edit("wrappers/llm_wrappers.py", '        elif control_args.decoder_model == "dory-4b":\n',
     '        elif control_args.decoder_model in ("dory-4b", "dory-tiny"):\n', '("dory-4b", "dory-tiny"')
# sharding: cut at the loop boundaries (model.py unrolls the loops with a copy of the weights each; a weight read by
# two ops in one context is turned into a DDR tensor by the HTP compiler, ~1 byte per parameter recomputed every step)
p = os.path.join(L, "wrappers/llm_wrappers.py")
s = open(p).read()
if '"shard_layers": [a.n_input, a.n_input + a.n_recurrent]' in s:   # a tree patched before the unrolling
    s = s.replace("                # input sublayers | the looped block, both loops | output sublayers and lm_head\n",
                  "                # input sublayers | each loop (unrolled, own weights) | output sublayers and lm_head\n")
    s = s.replace('"shard_layers": [a.n_input, a.n_input + a.n_recurrent]',
                  '"shard_layers": [a.n_input + k * a.n_recurrent for k in range(a.n_loops + 1)]')
    open(p, "w").write(s)
    print("patched wrappers/llm_wrappers.py (unrolled sharding)")
edit("wrappers/llm_wrappers.py",
     "            SplitGraph, setting = model_sharding.get_split_graph_pass(\n"
     "                self.meta[\"get_n_layers\"],\n"
     "                shares=self.config.num_sharding,\n"
     "                pattern=self._get_sharding_get_pattern(),\n"
     "                layer_prefix_offsets=layer_prefix_offsets,\n"
     "            )\n",
     "            SplitGraph, setting = model_sharding.get_split_graph_pass(\n"
     "                self.meta[\"get_n_layers\"],\n"
     "                shares=self.config.num_sharding,\n"
     "                pattern=self._get_sharding_get_pattern(),\n"
     "                layer_prefix_offsets=layer_prefix_offsets,\n"
     "            )\n"
     "            if self.control_args.decoder_model in (\"dory-4b\", \"dory-tiny\"):\n"
     "                # input sublayers | each loop (unrolled, own weights) | output sublayers and lm_head\n"
     "                a = self.model_args\n"
     "                setting[QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY] = {\n"
     "                    \"shard_layers\": [a.n_input + k * a.n_recurrent for k in range(a.n_loops + 1)],\n"
     "                    \"pattern\": r\"^sublayers\\.(\\d+)\",\n"
     "                }\n",
     '"shard_layers": [a.n_input + k * a.n_recurrent for k in range(a.n_loops + 1)]')
p = os.path.join(L, "__init__.py")
s = open(p).read()
shard_line = "    num_sharding = 4   # input | loop 1 | loop 2 | output (llm_wrappers cuts at the loop boundaries)\n"
for cls in ("class Dory_4B(LLMModelConfig):", "class Dory_Tiny(LLMModelConfig):"):
    i = s.index(cls)
    end = s.index("quant_recipe = DoryQuantRecipe", i)
    for old in ("    num_sharding = 1\n",
                "    num_sharding = 3   # input | looped block | output (llm_wrappers cuts outside the loop)\n"):
        j = s.find(old, i)
        if 0 <= j < end:
            s = s[:j] + shard_line + s[j + len(old):]
            print("patched __init__.py num_sharding", cls)
open(p, "w").write(s)
# Dory's shared cache layout (dory/model.py cache_slots): the KV inputs are found by the shape of their concat
# (cache slots + AR), the masks are [B, 1, AR, slots + AR] for the full and the ring caches
edit("wrappers/llm_wrappers.py",
     "        else:\n"
     "            self.kv_cache_shape = {\n"
     "                # single head, kv input\n",
     "        elif self.control_args.decoder_model in (\"dory-4b\", \"dory-tiny\"):\n"
     "            hd, ar = self.meta[\"get_head_dim\"], self.meta[\"get_ar_len\"]\n"
     "            ctx, ring = self.meta[\"get_max_context_len\"], self.meta[\"get_ring_slots\"]\n"
     "            cols = {ctx} if ar == ctx else {ctx + ar, ring + ar}\n"
     "            self.kv_cache_shape = (\n"
     "                {(hd, c) for c in cols} | {(c, hd) for c in cols} | {(hd, ar), (ar, hd)}\n"
     "            )\n"
     "        else:\n"
     "            self.kv_cache_shape = {\n"
     "                # single head, kv input\n",
     "            cols = {ctx} if ar == ctx else {ctx + ar, ring + ar}")
edit("wrappers/llm_wrappers.py",
     "        freq_shape = {\n"
     "            (self.meta[\"get_ar_len\"], self.meta[\"get_head_dim\"] // 2),\n"
     "        }\n",
     "        if self.control_args.decoder_model in (\"dory-4b\", \"dory-tiny\"):\n"
     "            b, ar = self.meta[\"get_max_batch_size\"], self.meta[\"get_ar_len\"]\n"
     "            ctx, ring = self.meta[\"get_max_context_len\"], self.meta[\"get_ring_slots\"]\n"
     "            if ar != ctx:\n"
     "                atten_mask_shape = {(b, 1, ar, ctx + ar), (b, 1, ar, ring + ar)}\n"
     "        freq_shape = {\n"
     "            (self.meta[\"get_ar_len\"], self.meta[\"get_head_dim\"] // 2),\n"
     "        }\n",
     "atten_mask_shape = {(b, 1, ar, ctx + ar), (b, 1, ar, ring + ar)}")
# calibration reuse (dory/calib_cache.py) and no 20+ GB QDQ file unless asked (XTG_SAVE_QDQ=1)
p = os.path.join(L, "wrappers/llm_wrappers.py")
s = open(p).read()
old_load = ("                    # an earlier calibration of the same model code: one forward fills the weight observers,\n"
            "                    # then the saved activation observers replace the ones this forward touched\n"
            "                    from executorch.examples.qualcomm.oss_scripts.dory.calib_cache import (\n"
            "                        load_activation_observers,\n"
            "                    )\n\n"
            "                    self.decoder(*self.export_input)\n")
if old_load in s:   # a tree patched before observe_weights
    s = s.replace(old_load,
                  "                    # an earlier calibration of the same model code: the weight observers see their weights\n"
                  "                    # (no forward: the calibration graph is AR = context, hours on the CPU at 32K), the saved\n"
                  "                    # activation observers go back in\n"
                  "                    from executorch.examples.qualcomm.oss_scripts.dory.calib_cache import (\n"
                  "                        load_activation_observers,\n"
                  "                        observe_weights,\n"
                  "                    )\n\n"
                  "                    observe_weights(self.decoder)\n")
    open(p, "w").write(s)
    print("patched wrappers/llm_wrappers.py (observe_weights)")
edit("wrappers/llm_wrappers.py",
     "                else:\n"
     "                    PTQStrategy(\n"
     "                        inference=self._decoder_inference,\n"
     "                        module=self.decoder,\n"
     "                        tok_embedding=self.tok_embedding,\n"
     "                        seq_mse_candidates=self.config.seq_mse_candidates,\n"
     "                    ).quantize(calib_loader=calibration_dataloaders)\n"
     "                    logging.info(\"Calibration complete\")\n",
     "                elif os.environ.get(\"XTG_CALIB_LOAD\"):\n"
     "                    # an earlier calibration of the same model code: the weight observers see their weights\n"
     "                    # (no forward: the calibration graph is AR = context, hours on the CPU at 32K), the saved\n"
     "                    # activation observers go back in\n"
     "                    from executorch.examples.qualcomm.oss_scripts.dory.calib_cache import (\n"
     "                        load_activation_observers,\n"
     "                        observe_weights,\n"
     "                    )\n\n"
     "                    observe_weights(self.decoder)\n"
     "                    load_activation_observers(self.decoder, os.environ[\"XTG_CALIB_LOAD\"])\n"
     "                    logging.info(\"Calibration loaded from %s\", os.environ[\"XTG_CALIB_LOAD\"])\n"
     "                else:\n"
     "                    PTQStrategy(\n"
     "                        inference=self._decoder_inference,\n"
     "                        module=self.decoder,\n"
     "                        tok_embedding=self.tok_embedding,\n"
     "                        seq_mse_candidates=self.config.seq_mse_candidates,\n"
     "                    ).quantize(calib_loader=calibration_dataloaders)\n"
     "                    logging.info(\"Calibration complete\")\n"
     "                    if os.environ.get(\"XTG_CALIB_SAVE\"):\n"
     "                        from executorch.examples.qualcomm.oss_scripts.dory.calib_cache import (\n"
     "                            save_activation_observers,\n"
     "                        )\n\n"
     "                        save_activation_observers(self.decoder, os.environ[\"XTG_CALIB_SAVE\"])\n",
     'os.environ.get("XTG_CALIB_LOAD")')
edit("wrappers/llm_wrappers.py",
     "            # Saving Decode QDQ Model EP for SQNR evaluation\n"
     "            qdq_ep = torch.export.export(\n",
     "            # Saving Decode QDQ Model EP for SQNR evaluation (tens of GiB for a 4B model: XTG_SAVE_QDQ=1)\n"
     "            qdq_ep = None if os.environ.get(\"XTG_SAVE_QDQ\", \"0\") != \"1\" else torch.export.export(\n",
     'XTG_SAVE_QDQ')
edit("wrappers/llm_wrappers.py",
     "            qdq_ep_path = f\"{self.decode.control_args.artifact}/{DECODE_QDQ_FILENAME}\"\n"
     "            torch.export.save(qdq_ep, qdq_ep_path)\n"
     "            logging.info(f\"QDQ EP saved to {qdq_ep_path}\")\n",
     "            qdq_ep_path = f\"{self.decode.control_args.artifact}/{DECODE_QDQ_FILENAME}\"\n"
     "            if qdq_ep is not None:\n"
     "                torch.export.save(qdq_ep, qdq_ep_path)\n"
     "                logging.info(f\"QDQ EP saved to {qdq_ep_path}\")\n",
     "            if qdq_ep is not None:\n                torch.export.save")
p = os.path.join(L, "wrappers/llm_wrappers.py")
s = open(p).read()
if "\nimport os\n" not in s:
    s = s.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)
    open(p, "w").write(s)
    print("patched llm_wrappers.py import os")
# a short calibration graph for long contexts (XTG_CALIB_CTX, with XTG_CALIB_LOAD): the calibration graph runs AR =
# its context, and at 96K its two masks alone are 77 GB; its nodes and encodings do not depend on the length.
# _encoding_override copies every parameter of the calibration graph into the prefill / decode graphs: a table sized
# by the context (RoPE's) keeps the target graph's own and takes only the encoding.
edit("wrappers/base_component.py",
     "    elif mode == Mode.CALIBRATE:\n"
     "        ar_len = control_args.max_context_len\n",
     "    elif mode == Mode.CALIBRATE:\n"
     "        ar_len = int(os.environ.get(\"XTG_CALIB_CTX\", control_args.max_context_len))\n",
     "XTG_CALIB_CTX")
edit("wrappers/base_component.py",
     "    model_args.max_seq_len = control_args.max_seq_len\n"
     "    model_args.max_context_len = control_args.max_context_len\n",
     "    model_args.max_seq_len = control_args.max_seq_len\n"
     "    model_args.max_context_len = control_args.max_context_len\n"
     "    if mode == Mode.CALIBRATE and ar_len != control_args.max_context_len:   # XTG_CALIB_CTX\n"
     "        model_args.max_seq_len = model_args.max_context_len = ar_len\n",
     "if mode == Mode.CALIBRATE and ar_len != control_args.max_context_len")
p = os.path.join(L, "wrappers/base_component.py")
s = open(p).read()
if "\nimport os\n" not in s:
    s = s.replace("\nimport math\n", "\nimport math\nimport os\n", 1)
    open(p, "w").write(s)
    print("patched base_component.py import os")
edit("wrappers/llm_wrappers.py",
     "            _set_attr(\n"
     "                unquantized_model,\n"
     "                unquantized_node.target,\n"
     "                _get_attr(quantized_model, quantized_node.target),\n"
     "            )\n",
     "            src = _get_attr(quantized_model, quantized_node.target)\n"
     "            dst = _get_attr(unquantized_model, unquantized_node.target)\n"
     "            # a table sized by the context (RoPE's) from a shorter calibration graph (XTG_CALIB_CTX): keep this\n"
     "            # graph's own, the encoding still comes over below\n"
     "            if not (isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape != dst.shape):\n"
     "                _set_attr(unquantized_model, unquantized_node.target, src)\n",
     "a table sized by the context (RoPE's)")
# "dory-4b-l1": Dory 4B with one pass of the looped block (about 30% fewer bytes per token, lower quality), from the
# offline Hugging Face repo local/dory-4b-l1 (config.json with n_recurrent_loops 1, the real weights linked:
# hf_local_variant.py); cut input | the loop | output
p = os.path.join(L, "wrappers/llm_wrappers.py")
s = open(p).read()
if 'in ("dory-4b", "dory-tiny")' in s:
    s = s.replace('in ("dory-4b", "dory-tiny")', 'in ("dory-4b", "dory-tiny", "dory-4b-l1")')
    open(p, "w").write(s)
    print("patched llm_wrappers.py (dory-4b-l1)")
edit("decoder_constants.py", '    "dory-tiny": "dory",\n', '    "dory-tiny": "dory",\n    "dory-4b-l1": "dory",\n',
     '"dory-4b-l1"')
edit("__init__.py", '    "dory-tiny": DoryModel,\n', '    "dory-tiny": DoryModel,\n    "dory-4b-l1": DoryModel,\n',
     '"dory-4b-l1": DoryModel')
p = os.path.join(L, "__init__.py")
s = open(p).read()
if '@register_llm_model("dory-4b-l1")' not in s:
    s = s.rstrip("\n") + '''


@register_llm_model("dory-4b-l1")
@dataclass(init=False, frozen=True)
class Dory_4B_L1(LLMModelConfig):
    repo_id: str = "local/dory-4b-l1"
    params_path: str = os.path.join(BASE_DIR, "../dory/config/4b_l1_config.json")
    convert_weights = convert_dory_weights
    transform_weight = False
    instruct_model = True
    num_sharding = 3   # input | the loop | output
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quant_recipe = DoryQuantRecipe
'''
    open(p, "w").write(s)
    print("patched __init__.py (dory-4b-l1)")
print("done")
