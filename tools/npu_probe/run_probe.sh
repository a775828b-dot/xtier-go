#!/system/bin/sh
# run_probe.sh [quick]: the NPU probe on the device (adb shell, from the kit directory). Writes results/ and
# results.tar.gz: environment (T0), per-model execute times with profiling logs and the first 32 output rows
# (T1 / T2 / T3), memory (T7), a 3-minute sustained run with temperatures (T4), the same run next to a GPU load
# when GPU_LOAD names a command (T5), and whether the app's domain can open the CDSP (T8, needs APP=<package>).
# quick: one model per test, 1-minute sustained run.
K=$(cd "$(dirname "$0")" && pwd)
cd "$K" || exit 1
export LD_LIBRARY_PATH="$K/lib"
export ADSP_LIBRARY_PATH="$K/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp"
R=results
rm -rf $R && mkdir -p $R/acc $R/prof
QUICK=$1
APP=${APP:-io.github.a775828b.xtiergo}
now() { cut -d' ' -f1 /proc/uptime; }
since() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.2f", b - a }'; }
temps() {  # zone=milli-degrees for the CPU / GPU / NSP / skin zones
  for z in /sys/class/thermal/thermal_zone*; do
    t=$(cat $z/type 2>/dev/null); case "$t" in *cpu*|*gpu*|*nsp*|*cdsp*|*skin*|*battery*|*sys-therm*) printf '%s=%s ' "$t" "$(cat $z/temp 2>/dev/null)";; esac
  done; echo
}
run() {  # run NAME LIST N [extra qnn-net-run flags]: N inferences with basic profiling, the first output kept
  n=$1; list=$2; count=$3; shift 3
  out=$R/out/$n
  t0=$(now)
  ./bin/qnn-net-run --backend lib/libQnnHtp.so --retrieve_context models/$n.bin --input_list $list --output_dir $out \
    --perf_profile burst --profiling_level basic --num_inferences $count --keep_num_outputs 1 --log_level warn "$@" \
    > $R/prof/$n.log 2>&1 &
  pid=$!
  hwm=""
  while kill -0 $pid 2>/dev/null; do
    s=$(grep -E 'VmHWM|VmRSS' /proc/$pid/status 2>/dev/null | tr -s ' \t' ' ' | tr '\n' ' ')
    [ -n "$s" ] && hwm=$s
    sleep 0.5
  done
  wait $pid
  rc=$?
  echo "$n rc=$rc wall_s=$(since $t0) n=$count $* mem: $hwm" >> $R/times.txt
  for p in $out/qnn-profiling-data*.log; do [ -f "$p" ] && ./bin/qnn-profile-viewer --input_log "$p" > $R/prof/$n.profile.txt 2>&1 && cp "$p" $R/prof/$n.qnnlog; done
  # the first 32 rows of every output (row widths from the shapes) for the accuracy check on the server
  mkdir -p $R/acc/$n
  for f in $out/Result_0/*.raw; do
    [ -f "$f" ] || continue
    b=$(basename $f .raw)
    case $b in h2|y) w=2560;; q|g2) w=4096;; k|v) w=1024;; *) w=0;; esac
    [ $w -gt 0 ] && dd if=$f of=$R/acc/$n/$b.raw bs=$((w * 4)) count=32 2>/dev/null
  done
  rm -rf $out
}

echo "== T0 environment" | tee $R/summary.txt
{
  echo "model=$(getprop ro.product.model) soc=$(getprop ro.soc.model) board=$(getprop ro.board.platform) android=$(getprop ro.build.version.release)"
  getprop ro.build.fingerprint
  uname -a
  ls -lZ /dev/fastrpc-* /dev/kgsl-3d0 2>&1
  grep -h -E 'cdsprpc|OpenCL' /vendor/etc/public.libraries.txt /system/etc/public.libraries.txt 2>&1
  temps
} > $R/env.txt 2>&1
./bin/qnn-platform-validator --backend dsp --coreVersion --libVersion --testBackend > $R/platform_validator.txt 2>&1
tail -5 $R/platform_validator.txt | tee -a $R/summary.txt

echo "== T8 app domain ($APP)" | tee -a $R/summary.txt
{
  run-as $APP id 2>&1
  for d in /dev/fastrpc-cdsp /dev/fastrpc-cdsp-secure /dev/kgsl-3d0; do
    run-as $APP sh -c "exec 3<$d && echo '$d open ok'" 2>&1
  done
} > $R/app_domain.txt 2>&1
cat $R/app_domain.txt | tee -a $R/summary.txt

echo "== T1-T3 execute times" | tee -a $R/summary.txt
if [ "$QUICK" = quick ]; then MODELS="tiny_m16_w8a16 seg_m1024_w4a16 seg_m1024_w8a8"; else
  MODELS=$(ls models | sed 's/\.bin$//' | sort); fi
for n in $MODELS; do
  m=${n#*_m}; m=${m%%_*}
  run $n list_m$m.txt 20
  tail -1 $R/times.txt | tee -a $R/summary.txt
done
run tiny_m16_w8a16 list_m16.txt 200
mv $R/prof/tiny_m16_w8a16.profile.txt $R/prof/tiny_m16_w8a16.n200.profile.txt 2>/dev/null
run tiny_m16_w8a16 list_m16.txt 200 --shared_buffer
mv $R/prof/tiny_m16_w8a16.profile.txt $R/prof/tiny_m16_w8a16.n200.shared.profile.txt 2>/dev/null
tail -2 $R/times.txt | tee -a $R/summary.txt

SUSTAIN=180; [ "$QUICK" = quick ] && SUSTAIN=60
echo "== T4 sustained seg_m1024_w4a16 ${SUSTAIN}s" | tee -a $R/summary.txt
temps > $R/thermal.txt
t0=$(now)
i=0
while [ "$(awk -v a=$t0 -v b=$(now) 'BEGIN { print (b - a < '$SUSTAIN') }')" = 1 ]; do
  i=$((i + 1))
  run seg_m1024_w4a16 list_m1024.txt 30
  mv $R/prof/seg_m1024_w4a16.profile.txt $R/prof/sustain_$i.profile.txt 2>/dev/null
  echo "t=$(since $t0) $(temps)" >> $R/thermal.txt
done
tail -3 $R/thermal.txt | cut -c1-300 | tee -a $R/summary.txt

if [ -n "$GPU_LOAD" ]; then
  echo "== T5 next to the GPU: $GPU_LOAD" | tee -a $R/summary.txt
  sh -c "while true; do $GPU_LOAD; done" > $R/gpu_load.txt 2>&1 &
  gp=$!
  sleep 3
  run seg_m1024_w4a16 list_m1024.txt 30
  mv $R/prof/seg_m1024_w4a16.profile.txt $R/prof/with_gpu.profile.txt 2>/dev/null
  sleep 8
  pkill -P $gp 2>/dev/null   # the GPU command first: once its loop is gone it would be re-parented
  kill $gp 2>/dev/null
  tail -1 $R/times.txt | tee -a $R/summary.txt
fi

tar -czf results.tar.gz $R
echo "== done: $K/results.tar.gz" | tee -a $R/summary.txt
