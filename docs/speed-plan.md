# 速度方案：DSH 32K 下的 prefill / decode

[回到文档目录](README.md)

2026-11-07 Y900 真机摸底之后定的方案。标"实测"的数字来自 Y900（Snapdragon 8 Elite：Adreno 830 GPU、Hexagon V79 NPU），
其余是按实测推的估算。测法和原始数据见[真机摸底报告](reports/2026-11-07-y900-npu-gpu-probe.md)。

## 目标和口径

- **任务**：DSH agent 任务，会话上下文涨到 32K。新内容（工具结果、读进来的文件）在它所在的深度 prefill，decode 在当前深度。
  DSH 先不接：现在用简单 agent 任务代替（`tools/agent_bench.py`：4 个会话，在本仓库源码上做小改动，起点 3K–28K）。
  下面按深度 4K / 16K / 32K 分开算，另给一个 0 → 32K 会话的平均。
- **目标**（11-08 按实测调整）：prefill 在 4K–8K 深度 250–300 tok/s、32K 深度 120–180；decode 在 4K 20–25 tok/s、32K 15–20。
  上下文保 128K（per_loop KV，投机解码因此是精确的）。原定 prefill 300 / decode 40：decode 卡在内存带宽上（2 循环每 token
  读 3.2 GiB，单 token 极限 4K 约 18、32K 约 15），agent 任务上提示查找草稿每次前向只出 1.9 个；32K 深度的 prefill 卡在
  GPU 注意力上（q·k 要 FP32）。再往上要训草稿头（S2）或改模型的跑法，暂不做。
- **精度**：int8 权重 / int8 激活（W8A16、W4A8）可以用，A/B 通过才能当默认：PPL、照抄、对 vLLM 的逐 token logprob、
  DSH 任务结果（[validation.md](validation.md)）。

## 物理账

Dory 4B 跑 2 次循环，每个 token 有 56 次注意力和 56 段线性层（"段" = 两次注意力之间的全部算子，每段 91.75 M 权重）。
56 次注意力里 14 次是全局的（第 6、14 … 70 层，循环段里的 5 层算两次），42 次是滑窗（窗口 1024）。KV 是 FP8，
每个位置、每次全局注意力 2 KiB。

| 每 token | 4K | 16K | 32K |
|---|---:|---:|---:|
| 线性层计算 | 10.3 GFLOP | 10.3 | 10.3 |
| 注意力计算 | 1.6 GFLOP | 4.5 | 8.2 |
| decode 读的字节（权重 + KV） | 3.2 + 0.2 GiB | 3.2 + 0.5 | 3.2 + 1.0 |

- **decode 过 20 只能靠投机解码。** 带宽按 60 GB/s 算（实测 66），单 token 的上限 4K 约 17 tok/s、32K 约 14。验证几个 token
  读的权重和 KV 跟 decode 一步一样多，每步接受 a 个就快约 a 倍，前提是验证步的内核真的只读一次。
- **prefill 到 300 要 NPU 和 GPU 同时干活。** 每 token 只有 3.3 ms：线性层要 3.1 TFLOP/s，32K 深度的注意力还要再加
  2.5 TFLOP/s，哪个单元都扛不下两样。q·k 必须 FP32 累加（[dory.md](dory.md#精度注意注意力很尖)），所以 GPU 做注意力
  的上限是 FP32 峰值 2.81 TFLOP/s。

## 实测摘要

### 现在的引擎

325 token 提示、生成 32 个、2 循环、last_loop、FP8 KV（main `8aaf43d`）：

| 后端 | prefill tok/s | decode tok/s |
|---|---:|---:|
| CPU（NEON，8 线程） | 7.7 | 4.4 |
| gpu-linear（只有矩阵乘在 GPU） | 19.9 | 3.1 |
| gpu（整步驻留 GPU） | 31.9 | 6.0 |

gpu 的输出和 CPU 逐 token 一致。App 里 DEV 测速 33.0 / 5.8：OpenCL 在 App 进程里能用，6 s 上线。

时间花在哪（`XTG_GPU_PROFILE=1`）：

- **prefill**（919 token 的照抄提示，两块）：注意力占 59%，每次 240 ms（一行一个 work-group 的内核）；矩阵乘占 39%，
  FP16 分块 0.52 TFLOP/s，只有 FP16 峰值的 9%。**8K 提示**：prefill 8.8 tok/s（15 分钟），注意力占 GPU 时间的 82%；
  decode 3.4 tok/s（注意力每次 2.0 ms）。
- **decode**（per_loop，深度约 1.1K）：每 token 225 ms = GEMV 122 ms（224 次，读权重 24 GB/s，只用了带宽的 37%）
  + 注意力 56 ms（每次 1.0 ms）+ lm_head 11 ms + 小内核 5 ms + 发射空隙约 30 ms。
- **投机解码**（同一照抄任务，spec_k 8）：每步接受 8.1 个，草稿 90% 被接受，decode 却只快 19%（4.42 → 5.26）。
  原因是 M = 9 的 GEMV 没复用权重，耗时是 M = 1 的 9 倍；多行注意力每次 6.0 ms，是单行的 6 倍。

GPU 本身（`tests/gpu_probe`）：读 66 GB/s；峰值 FP32 2.81、FP16 5.61 TFLOP/s，int8 点积 8.78 TOP/s；异步发射每个 kernel
11.8 µs；GEMV M = 2 / 16 的耗时是 M = 1 的 1.9 / 15 倍。

### NPU

测的单位是一段。真实 NVFP4 权重解成 FP32，交给 QAIRT 2.47 量化，再和 float64 参考比误差（前 32 行的相对 RMS）。
时间是 qnn-net-run 每次 execute 的平均（basic profile 的 NetRun，含 FP32 输入输出的转换）。

图怎么写，差别最大（W8A16，M = 1024）：

| 写法 | 每段 | 误差 h2 / k / q |
|---|---:|---|
| 照写（归一化是 [M, C] / [M, 1] 的 Div） | 369 ms | 0.56 / 0.87 / 1.29% |
| 归一化改成每行一个倒数再乘 | 189 ms | 0.54 / 0.85 / 1.26% |
| **折叠** | **66.9 ms** | 0.57 / 0.87 / 1.29% |

折叠的依据：对每行一个正标量 c，norm(c·v) = norm(v)。o_proj 和 down 的输入在后面还要归一化，所以它们前面的两个归一化
可以去掉；slerp 的输入 x、h 本来就是单位向量，又去掉两个；SwiGLU 的两个缩放向量并进 gate / up 的权重行。68 个算子变成 42 个，
见 `tools/npu_probe/make_models.py` 的 `segment_onnx`。引擎里的 GPU 内核也可以照这样省。

量化方式（折叠后）：

| | 每段 | 误差 h2 / k / q | 结论 |
|---|---:|---|---|
| FP16（M = 1024） | 159.7 ms | 0.07 / 0.06 / 0.07% | 准，但慢 |
| **W8A16**（每行 INT8 权重，16 位激活，M = 1024） | **66.9 ms** | 0.57 / 0.87 / 1.29% | **选它**，A/B 待做 |
| W8A8（M = 1024） | 44.5 ms | 27 / 23 / 33% | 不能用：激活是离线标定、每张量一个的静态量化 |
| W4A16（M = 2048） | 140.9 ms | 8.0 / 15.8 / 23% | 不能用：每行 INT4；分块的 INT4（LPBQ）第一次试缩放全是 0，还没修 |

块大小（折叠 W8A16）：

| M | 512 | 768 | 1024 | 1536 | 2048 |
|---|---:|---:|---:|---:|---:|
| 每段 | 32.4 ms | 47.9 | 66.2 | 108.7 | 165.2 |
| 其中在 NPU 上 | 28.1 ms | 41.6 | 58.7 | 98.8 | 152.5 |
| 折算成线性层 tok/s | 283 | **286** | 276 | 252 | 221 |

- **剩下的时间**：矩阵乘和逐元素运算（归一化、slerp、SiLU）大约各占一半。转回 FP32 输出占 11%；NetRun 比 NPU 时间多出的
  约 6 ms 是 FP32 输入输出的转换和拷贝。接进引擎用原生 16 位张量后，这两项都能省掉，768 一块估计约 2.8 ms/token（约 357 tok/s）。
- **和 GPU 同时跑**：NPU 每段 47.0 → 44.0 ms，不受影响；GPU 矩阵乘 539 → 540 GFLOP/s，也不受影响；GPU 读带宽 66 → 55.7 GB/s（−16%）。
- **持续负载**：连续 4 分钟 2500 次，每段 47.3–47.5 ms 不变，NPU 42–44 °C。不过 qnn-net-run 在两次 execute 之间要读输入文件，
  NPU 实际只忙了一半时间；满负荷的长测等接进引擎后再做。
- **App 里能用 NPU（N0 已通过）**：App 进程（untrusted_app）自己打不开 /dev/fastrpc-cdsp（策略只给 App ioctl / read），
  libcdsprpc.so 会转去 vendor 的 dspservice 拿设备的 fd。条件是清单声明 `<uses-native-library android:name="libcdsprpc.so">`；
  另外，App 拉起的程序要把 `/vendor/lib64` 放进 LD_LIBRARY_PATH。App 里跑 seg_m768_w8a16：每段 NetRun 46 ms、NPU 上 41 ms，
  和 adb 下一样，输出逐字节相同。
- 固定开销：2560 × 2560、M = 16 的小图，每次 1.12 ms（NPU 上 0.48 ms）。

## 结论

1. **prefill 靠 NPU 的矩阵乘。** 折叠后的 W8A16、768 一块，线性层已经 286 tok/s；GPU 的 FP16 矩阵乘只有 0.52 TFLOP/s，
   线性层约 50 tok/s。GPU 的 int8 点积做好了，线性层也能到两三百，但注意力也在 GPU 上，两样只能串着跑。
2. **注意力两处都要重写**：prefill 的（现在占 59%）和验证步的多行注意力（单行的 6 倍）。深度一大，prefill 卡在 GPU 注意力上。
3. **decode 先修内核再谈投机解码。** GEMV 只用了带宽的 37%，发射空隙占 13%，M > 1 又不复用权重。这三项修好，单 token
  4K 约 17、32K 约 14，投机解码再乘每次前向出的 token 数。
4. **提示查找草稿在 agent 任务上只到每次前向约 1.9–2.0 个 token**（S1，见下）。思考占生成量的 80%，每次只能出 1.8 个左右，
  草稿规则改成最长匹配、一次验证 3 条候选，也只到 2.1。decode 40 要约 2.6 个（4K）、3.2 个（32K），只靠查找到不了，
  要换成模型做草稿。
5. **decode 留在 GPU。** NPU 用的是 INT8 权重，每步要读的字节是 NVFP4 的 1.8 倍。
6. **内存**：NPU 要另放一份 INT8 权重，约 3.2 GiB。32K 时全部加起来约 7.0 GiB（GPU 权重 2.4 + NPU 3.2 + KV 1.0 + 临时 0.5），
   在 8 GiB 预算内；128K 约 9.6 GiB，超了。NPU 权重要降到 4 bit（约 1.7 GiB），或者长上下文时 NPU 不常驻。

## 计划

两条线可以并行：decode 在 GPU 上做，prefill 是 NPU 加 GPU 注意力。每一步都有门槛，没过不合并；改解码路径的，照旧先过
照抄和逐 token 检查。

### 先做（半天到一天，决定后面怎么走）

| 步 | 内容 | 通过 | 不通过 |
|---|---|---|---|
| N0 | App 进程里跑 NPU：APK 带 libQnnHtp 和 V79 stub，skel 放原生库目录并设 ADSP_LIBRARY_PATH；tiny_m16 和 seg_m768_w8a16 各跑一次，输出和 adb 下一致 | **11-07 夜通过**：走路线 N（NPU 线性层 + GPU 注意力） | 走路线 G：G6 提前，GPU int8 点积做线性层；prefill 目标降为会话平均约 150 |
| S1 | 先不接 DSH，用简单 agent 任务（`tools/agent_bench.py` 的 4 个会话）统计提示查找草稿每次前向能出几个 token（默认温度、贪心、关思考各一组，思考单独统计） | 4K ≥ 2.6、32K ≥ 3.2：decode 40 可达 | **11-07 夜：1.9（k = 8）/ 2.0（k = 16）**，没达到。decode 按 25–30 算，要 40 得做 S2 |
| S2 | （研究，11-08 定暂不做）模型做草稿：EAGLE 式的小草稿头，用 Dory 自己的隐状态训练；词表要缩小，否则每个草稿 token 都要读一遍 0.3–0.6 GiB 的 lm_head | 默认温度、开思考时每次前向 ≥ 2.6 个 | — |

### decode（GPU）

| 步 | 内容 | 门槛 |
|---|---|---|
| G1 | GEMV 跑满带宽：权重加载时重排（量化码和缩放分开，按 work-group 的块连续存）；16 字节向量读，每个线程同时有多个读在路上；image1d_buffer 走纹理缓存；subgroup 规约 | gpu_probe M = 1 ≥ 55 GB/s；decode 4K ≥ 11 tok/s。**11-07 深夜第一步**：权重改 k-major（[K/32][行] 的 16 字节块），一行一个线程、K 切 2 份（K ≤ 3072，走 image）或 8 份。引擎里 GEMV 每 token 119 → 86 ms（约 34 GB/s；探针里同一矩阵连跑能到 47–50，有缓存命中），325 token decode 6.0 → 6.8 tok/s，服务器 20 → 40。GPU 快了以后主机发射开销（每 token 约 28 ms）露出来，G4 要提前 |
| G2 | 小 M 复用权重（M = 2–16）：每个权重只解码一次，对 M 行做乘加，激活放 local memory | M = 8 / 16 的耗时 ≤ M = 1 的 1.5 / 2 倍 |
| G3 | 验证步的多行注意力：split-K 内核扩成 R 行，一个 KV 头的 4 个 q 头 × R 行共用一次 KV 读 | R = 8 的耗时 ≤ 单行的 1.3 倍；照抄任务 spec_k 8 的 decode ≥ spec_k 0 的 3 倍 |
| G4 | 发射开销：用 `cl_qcom_recordable_queues` 把 decode 一步录下来，以后每步只改参数重放；RoPE、归一化、写 KV 等小内核合并 | GPU 空闲空隙 < 5%（现在 13%） |
| G5 | lm_head BF16 → FP8（每行一个缩放），每 token 少读 0.31 GiB | PPL 差 < 0.1%，照抄一致 |

### prefill（NPU + GPU）

| 步 | 内容 | 门槛 |
|---|---|---|
| N1 | NPU 线性层接进引擎：QNN HTP 运行时随 APK；38 个段的上下文二进制（折叠、W8A16、M = 768，尾块补齐或另备 M = 256 的图）；段的输入输出用原生 16 位张量（rpcmem / dmabuf，GPU 用 `cl_qcom_dmabuf_host_ptr` 零拷贝），中间不回 CPU | 和 GPU 路径比 prefill 后的 logits（top-1 一致率、KL）；PPL 差 < 0.5%；线性层 ≥ 330 tok/s |
| N2 | GPU prefill 注意力（两条路线都要）：flash-attention 式分块，FP8 K / V 在内核里解码，q·k FP32 累加、在线 softmax、P·V 用 FP16；一个 KV 头的 4 个 q 头共用 K / V | 325 / 919 / 2738 token 提示的贪心输出和现在一致；16K 深度 ≥ 1.5 TFLOP/s |
| N3 | NPU / GPU 流水：块 i 的注意力在 GPU 上算时，块 i+1 的段在 NPU 上算。按层错开即可：块 i+1 第 l 层的注意力只用到块 i 第 l 层的 K / V | 同时跑的速度 ≥ 单独跑时较慢那一边的 85% |
| N4 | NPU 上剩下的逐元素运算：nGPT 的归一化 + slerp 写成 HVX 自定义算子（QNN op package），或 MLP 中间激活改 8 位（要 A/B） | 768 一块每段 ≤ 30 ms |
| G6 | GPU int8 点积矩阵乘（W4A8）：E2M1 × 2 是不超过 12 的整数，权重放进 int8 不丢精度；激活每 16 / 32 个一组动态量化；点积后乘块缩放，进 FP32 累加。路线 G 的主力，路线 N 的备用 | gpu_probe ≥ 3 TOP/s；A/B 通过 |
| N5 | （研究）把一部分注意力挪到 NPU：滑窗层形状固定，最好做；全局层要按 KV 长度分桶的静态图，外加一份 FP16 的 K / V | 32K 深度的 prefill 有提升才继续 |

### 前缀缓存（P1，小改动）

客户端不回传思考（`reasoning_content`）时，新提示在上一轮的生成提示里就分叉了（`<think>\n` 对 `<think></think>`），比滑窗快照
早一个 token，于是每一轮都从零 prefill（agent_bench 第一版实测每步 cached 0）。改法：恢复快照后再回退最多 kRewindSlack
个位置。DSH 会回传思考，不受影响；`tools/agent_bench.py` 也已改成回传。

### 内存（M1，和 N1 一起做）

GPU 权重只放一份：映射模型文件或用 dmabuf（现在是 CPU 映射 + GPU 副本）。NPU 的上下文二进制约 3.2 GiB，128K 时二选一：
修好 LPBQ 降到 4 bit，或者 64K 以上卸载 NPU、退回 GPU prefill。门槛：128K 会话不触发 LMK，记下峰值内存。

## 预计（估算）

decode，G1–G5 做完后，每次前向出 a 个 token（验证步按单步的 1.1 倍算）：

| a | 1（不投机） | 1.9（S1 实测） | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 4K | 17 | 30 | 31 | 47 | 63 |
| 32K | 14 | 24 | 25 | 38 | 50 |

decode 40 要每次前向出 4K 约 2.6 个、32K 约 3.2 个。简单 agent 任务实测（S1）：默认温度、开思考 1.90（k = 8），贪心 2.11，
关思考 2.09；照抄类内容 8.1。所以 decode 40 要等 S2 的草稿头。

prefill（NPU 线性层 2.8 ms/token，GPU 注意力 1.5 TFLOP/s，流水效率 85%）：

| | 4K | 16K | 32K | 0 → 32K 会话平均 |
|---|---:|---:|---:|---:|
| 现在 | 32（325 token） | — | — | — |
| 只做 N2（矩阵乘还是现在的 FP16） | 47 | 44 | 39 | 44 |
| 路线 G：G6（3 TOP/s）+ N2 | 212 | 152 | 110 | 152 |
| 路线 N：N1 + N2，串行 | 244 | 167 | 118 | 168 |
| 路线 N：再加 N3 流水 | 304 | 268 | 150 | 234 |
| 再加 N4，注意力到 1.9 TFLOP/s | 386 | 333 | 188 | 295 |

所以 prefill 300：4K–16K 深度靠 N1–N4 能到；会话平均要 N4 加上更快的注意力。32K 深度这一个点，注意力每 token 8.2 GFLOP，
跑满峰值也要约 2.4 ms，按一半效率是 4–5 ms，单靠 GPU 到不了 300。要靠 N5，或者把 q·k 改成"FP16 分段累加，再并入 FP32"
（要先过 Dory 尖注意力的精度对照）。

## 风险

- **别的手机上的 NPU**：N0 只在 Y900 上验证过（经 dspservice 拿设备），别家的系统策略可能不同。App 要保留 NPU 自检，
  不通过就退回 GPU（路线 G，prefill 会话平均约 150）。
- **W8A16 的 A/B 不过**：只能退回 NPU FP16，线性层约 115 tok/s，或者做混合精度（敏感的矩阵乘用 FP16）。
- **注意力内核的效率**：上表按 FP32 / FP16 峰值的一半估算，Adreno 上没有现成的同类内核可以对照。
- **投机解码的接受数**：S1 已测，提示查找只到 1.9–2.0，目标因此定在 decode 20–25（4K）/ 15–20（32K）。验证步在 Adreno 上
  如果算不到单步的 1.2 倍以内（FP32 算力有限），还要往下打折。
- **持续负载**：只测了 4 分钟、NPU 半负荷；32K 的提示要连续 prefill 两分钟左右，满负荷时会不会降频没测过。
- **128K 内存**：NPU 4 bit 方案（LPBQ）还没跑通过。

## 复现

- GPU：`adb shell /data/local/tmp/xtg/bin/gpu_probe`；引擎：`XTG_GPU_PROFILE=1 xtg-cli generate ... --opt backend=gpu`
  （[build.md](build.md) 的 adb shell 一节）。
- NPU（服务器上需要 QAIRT 2.47）：`tools/npu_probe/make_models.py`（默认折叠，`--plain` 照写）→ `build_kit.sh` →
  把 kit 推到设备上跑 `run_probe.sh` → 结果拿回服务器跑 `analyze.py device`。各脚本开头有说明。
- 简单 agent 任务：`tools/agent_bench.py LABEL --url http://127.0.0.1:18400`（对着 xtg-server 跑，测 prefill / decode 和投机解码
  计数）。草稿接受数：`tools/draft_sim.py trace`（对着返回 token id 的服务器生成同样的会话，比如 vLLM）→ `draft_sim.py sim
  [--whatif]`；`draft_sim.py check` 先核对模拟和引擎一致（11-07：同一个照抄请求，引擎 49 步 / 草稿 387 / 接受 349，模拟 49 / 392 / 350，
  差别来自重新分词）。
