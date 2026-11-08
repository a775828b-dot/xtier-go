#!/bin/bash
# export_dory.sh CTX OUT [DECODE_AR] [PREFILL_AR]: one context bucket of Dory 4B for the Snapdragon 8 Elite (SM8750)
# NPU: a .pte with prefill_forward (PREFILL_AR rows, default 64) and kv_forward (DECODE_AR rows, default 8: the verify
# step of a speculative decoder; 1 gives the plain hybrid export).
#
# Needs ET_ROOT (ExecuTorch v1.5.1 with tools/npu_et applied, see docs/npu.md), QNN_SDK_ROOT (QAIRT 2.37),
# ANDROID_NDK_ROOT, and ExecuTorch's Python environment active. The weights come from the BF16 release
# (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, through the Hugging Face cache). MODEL: dory-4b (default) or
# dory-4b-l1 (one pass of the loop; its offline repo comes from hf_local_variant.py). CALIB (default
# calib/MODEL.pt) is read when it exists and written after calibrating when it does not (calibrate at 4K first);
# CALIB_SAMPLES are the calibration prompts (make_calib.py). Past 8K the calibration graph stays at 4K
# (XTG_CALIB_CTX): its encodings do not depend on the length.
set -euo pipefail
CTX=${1:?context length}
OUT=$(realpath -m "${2:?output directory}")
DAR=${3:-8}
AR=${4:-64}
MODEL=${MODEL:-dory-4b}
CALIB=$(realpath -m "${CALIB:-calib/$MODEL.pt}")
CALIB_SAMPLES=$(realpath "${CALIB_SAMPLES:-calib.json}")
mkdir -p "$OUT" "$(dirname "$CALIB")"
if [ -f "$CALIB" ]; then export XTG_CALIB_LOAD=$CALIB; else export XTG_CALIB_SAVE=$CALIB; fi
if [ -f "$CALIB" ] && [ "$CTX" -gt 8192 ]; then export XTG_CALIB_CTX=4096; fi
if [ "$DAR" -gt 1 ]; then
  MODE=(--model_mode lookahead --ngram 2 --window "$DAR" --gcap 0)   # sizes kv_forward to DAR rows
else
  MODE=(--model_mode hybrid)
fi
cd "$ET_ROOT"
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --soc_model SM8750 --compile_only \
  --decoder_model "$MODEL" "${MODE[@]}" --prefill_ar_len "$AR" --max_seq_len "$CTX" --max_context_len "$CTX" \
  --temperature 0 --prompt "x" --calib_samples "$CALIB_SAMPLES" --artifact "$OUT"
ls -la "$OUT"/*.pte
