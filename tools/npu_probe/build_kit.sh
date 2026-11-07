#!/bin/bash
# build_kit.sh WORK KIT: QAIRT DLCs, HTP v79 (SM8750) context binaries and the device kit for the NPU probe.
#
#   WORK: make_models.py's output (onnx/, inputs/, calib_m*.txt, list_m*.txt, manifest.json)
#   KIT:  the directory to push to the device (bin/ lib/ dsp/ models/ inputs/ run_probe.sh)
#   env:  QNN_SDK_ROOT (QAIRT 2.47; bin/envsetup.sh sourced, its venv active), JOBS (parallel jobs, default 6),
#         MATRIX / ACCURACY (graph:variant lists, defaults below)
#
# Variants: fp16 = the FP32 graph (HTP runs it in FP16), wXaY = qairt-quantizer with X-bit per-row weights and
# Y-bit activations calibrated on calib_m<M>.txt. The device matrix is MATRIX; ACCURACY lists DLCs that are only
# run in the HTP x86 emulator here (tools/npu_probe/analyze.py compares both with ref.npz). The defaults are the
# 2026-11-07 device picks: W8A16 is the accurate variant, M = 768 the fastest chunk, w8a8 the speed reference.
set -euo pipefail
WORK=$(readlink -f "$1")
KIT=$(readlink -f "$2")
JOBS=${JOBS:-6}
L=$QNN_SDK_ROOT/lib/x86_64-linux-clang
MATRIX=${MATRIX:-"seg_m768:w8a16 seg_m768:w8a8 seg_m512:w8a16 seg_m1024:w8a16 seg_m1024:fp16 tiny_m16:w8a16 tiny_m16:fp16"}
ACCURACY=${ACCURACY:-"seg_m512:fp16 seg_m512:w8a16 seg_m512:w4a8"}
mkdir -p "$WORK/dlc" "$WORK/logs" "$WORK/ctx" "$KIT"/{bin,lib,dsp,models,inputs}

# 1. FP32 DLC per graph
for g in $(for c in $MATRIX $ACCURACY; do echo "${c%%:*}"; done | sort -u); do
  [ -s "$WORK/dlc/$g.dlc" ] || qairt-converter -i "$WORK/onnx/$g.onnx" -o "$WORK/dlc/$g.dlc" > "$WORK/logs/conv_$g.log" 2>&1 &
done
wait

variant() {  # variant GRAPH VAR [ctx]: the DLC (and with ctx the context binary) for one cell
  local g=$1 v=$2 want_ctx=${3:-} dlc m w a
  m=${g##*_m}
  if [ "$v" = fp16 ]; then dlc=$WORK/dlc/$g.dlc; else
    dlc=$WORK/dlc/${g}_$v.dlc
    w=${v#w}; w=${w%a*}; a=${v#*a}
    [ -s "$dlc" ] || qairt-quantizer --input_dlc "$WORK/dlc/$g.dlc" --output_dlc "$dlc" --input_list "$WORK/calib_m$m.txt" \
      --weights_bitwidth "$w" --act_bitwidth "$a" --bias_bitwidth 32 --use_per_row_quantization > "$WORK/logs/quant_${g}_$v.log" 2>&1
  fi
  [ -n "$want_ctx" ] || return 0
  cat > "$WORK/ctx/${g}_$v.v79.json" <<EOF
{"graphs": [{"vtcm_mb": 8, "O": 3, "graph_names": ["$g"]}],
 "devices": [{"dsp_arch": "v79", "soc_model": 69, "pd_session": "unsigned", "device_id": 0}]}
EOF
  cat > "$WORK/ctx/${g}_$v.json" <<EOF
{"backend_extensions": {"shared_library_path": "$L/libQnnHtpNetRunExtensions.so", "config_file_path": "$WORK/ctx/${g}_$v.v79.json"}}
EOF
  [ -s "$KIT/models/${g}_$v.bin" ] || qnn-context-binary-generator --backend "$L/libQnnHtp.so" --model "$L/libQnnModelDlc.so" \
    --dlc_path "$dlc" --binary_file "${g}_$v" --output_dir "$KIT/models" --config_file "$WORK/ctx/${g}_$v.json" \
    > "$WORK/logs/ctx_${g}_$v.log" 2>&1
  echo "$(date +%T) ${g}_$v ok"
}
export -f variant
export WORK KIT L

# 2. quantized DLCs and context binaries, JOBS at a time
{ for c in $MATRIX; do echo "${c%%:*} ${c##*:} ctx"; done; for c in $ACCURACY; do echo "${c%%:*} ${c##*:}"; done; } |
  xargs -P "$JOBS" -L 1 bash -c 'variant "$@" || echo "FAILED $*"' _

# 3. the kit: QNN runtime for aarch64-android, the v79 DSP side, inputs, lists, manifest and the run script
B=$QNN_SDK_ROOT/bin/aarch64-android
A=$QNN_SDK_ROOT/lib/aarch64-android
cp "$B"/{qnn-net-run,qnn-platform-validator,qnn-throughput-net-run,qnn-profile-viewer} "$KIT/bin/"
cp "$A"/{libQnnHtp.so,libQnnHtpV79Stub.so,libQnnSystem.so,libQnnHtpNetRunExtensions.so,libQnnHtpV79CalculatorStub.so,libPlatformValidatorShared.so,libQnnHtpProfilingReader.so,libQnnJsonProfilingReader.so} "$KIT/lib/"
cp "$QNN_SDK_ROOT"/lib/hexagon-v79/unsigned/{libQnnHtpV79Skel.so,libQnnHtpV79.so,libCalculator_skel.so} "$KIT/dsp/"
cp "$WORK"/inputs/*.raw "$KIT/inputs/"
cp "$WORK"/list_m*.txt "$WORK/manifest.json" "$KIT/"
cp "$(dirname "$(readlink -f "$0")")/run_probe.sh" "$KIT/"
du -sh "$KIT"
