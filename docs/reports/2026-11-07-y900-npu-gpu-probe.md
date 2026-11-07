# Y900 真机摸底：GPU 驻留路径、内核剖析、NPU、App 里的 NPU、草稿接受数

2026-11-07 晚，联想 Y900 经 USB 接到开发机，命令行工具和 NPU 包用 `adb shell` 跑；模型、上下文二进制在 x86 服务器上生成。
据此定的方案见 [speed-plan.md](../speed-plan.md)。原始输出在 [data/y900-20261107-probe/](data/y900-20261107-probe/)。

## 环境

| 项 | 值 |
|---|---|
| 设备 | Lenovo TB522FU（Y900），SM8750P，15 GiB 内存，Android 16（BQ2A.250831.001），内核 6.6.82 |
| GPU | Adreno 830v2，OpenCL 3.0，驱动 0800.72，12 个 CU，local memory 32 KiB，全局 7.5 GiB，单块上限 1 GiB |
| NPU | Hexagon V79（HTP），QAIRT / QNN 2.47.0，unsigned PD，6 个 HVX 线程 |
| 代码 | main `8aaf43d`，加上 `rope_norm` 的空指针修复（本报告同一批提交） |
| 模型 | Dory 1.0 4B NVFP4，2 循环 |

## GPU（`tests/gpu_probe`）

| 项 | 结果 |
|---|---:|
| 发射：异步（2000 个、最后一次 clFinish）/ 每个都 clFinish / 原来的往返（上传 + kernel + 读回） | 11.8 / 139 / 174 µs |
| 读带宽（设备缓冲区 / host_ptr） | 66.0 / 66.0 GB/s |
| 峰值 FP32 / FP16 / int8 点积 | 2.81 / 5.61 TFLOP/s / 8.78 TOP/s |
| gemv4（NVFP4，2560 × 7168）M = 1 / 2 / 4 / 8 / 16 | 0.424 / 0.801 / 1.602 / 3.246 / 6.398 ms（M = 1 时 24.3 GB/s） |
| FP16 分块矩阵乘 M = 512 / 1024，N = 7168，K = 2560 | 513 / 520 GFLOP/s |

扩展：有 `cl_qcom_recordable_queues`、`cl_qcom_dot_product8`、`cl_khr_integer_dot_product`、`cl_qcom_ml_ops`、
`cl_qcom_dmabuf_host_ptr`、`cl_qcom_onchip_global_memory`、`cl_khr_subgroups`；没有 `cl_khr_command_buffer`、
`cl_qcom_reqd_sub_group_size`。

## 引擎

325 token 提示（[generate-ids](data/y900-20261107/generate-ids.json)）、生成 32 个、8 线程、2 循环、last_loop、FP8 KV、
上下文 4096、prefill 分块 512：

| 后端 | prefill tok/s | decode tok/s | 输出 |
|---|---:|---:|---|
| cpu | 7.73 | 4.43 | 基准 |
| gpu-linear | 19.89 | 3.11 | 与 CPU 一致 |
| gpu（整步驻留） | 31.85 | 5.98 | 与 CPU 一致 |

同一后端（gpu、per_loop、上下文 16384）的 8000 token 提示（本仓库源码）：prefill 8.81 tok/s，decode 3.41 tok/s。GPU 时间里
`attend` 755 s（82.4%，896 次）、`gemmt` 150 s（16.4%）；decode 的 `attend_split` 每次 2.0 ms（约 1.1K 深度时 1.0 ms）。

第一次跑 gpu 在驱动里崩溃：k 的 `rope_norm` 调用把一个不用的缓冲区参数传成 NULL，Adreno 驱动不接受。改成传一个真实缓冲区后正常。

App（`.dev` 并排开发版）里选 GPU：6 s 上线，DEV 测速 prefill 33.0、decode 5.8 tok/s，说明 App 进程里能加载 OpenCL。

### 内核剖析（`XTG_GPU_PROFILE=1`）

`xtg-server`、per_loop、上下文 8192，同一个 919 token 的照抄请求（复制一段 C++ 代码），生成 400 个：

| | spec_k 0 | spec_k 8 |
|---|---:|---:|
| prefill tok/s | 20.21 | 20.21 |
| decode tok/s | 4.42 | 5.26 |
| 验证步 / 草稿 / 接受 | — | 49 / 387 / 349 |

prefill（两块）45.5 s：`attend`（一行一个 work-group）26.9 s，112 次，每次 240 ms；`gemmt` 17.8 s，896 次。

decode 每 token（spec_k 0）：

| kernel | 每 token | 说明 |
|---|---:|---|
| gemv4 | 122 ms | 224 次，每次 0.55 ms |
| attend_split + merge | 57 ms | 56 次，每次 1.0 ms（深度约 1.1K） |
| gemv16（lm_head，BF16） | 11.3 ms | — |
| 其余小内核 | 5 ms | rope_norm、slerp、swiglu_norm、gate_norm、kv_store |
| 合计 | 195 ms | 实际 225 ms：其余约 30 ms 是发射和同步的空隙 |

spec_k 8 的验证步（M = 9）：gemvm 每次 4.95 ms，是 gemv4 的 9 倍；多行 `attend` 每次 6.0 ms（单行 1.0 ms）；
lm_head 每步 44.5 ms。每步虽然接受了 8.1 个 token，一步却要 1.55 s（49 步共 75.9 s）。

## NPU

### 测法

一段是两次注意力之间的全部算子：注意力输出门、o_proj、slerp、MLP（gate / up 合成一个矩阵、SiLU 和 suv 缩放、归一化、
down）、slerp、下一层的 q / k / v / g。层取第 32–34 子层，权重是真实的 NVFP4 精确解码成 FP32；输入取自服务器上的一段真实
prefill。`make_models.py` 写 ONNX 和 float64 参考，`build_kit.sh` 用 QAIRT 量化，并生成 v79（soc_model 69）的上下文二进制。
设备上 `qnn-net-run --perf_profile burst` 每个模型跑 20 次（basic profile），时间取 Average 一节的 NetRun（含 FP32 输入输出的
转换和拷贝）和 Accelerator 时间；误差是前 32 行对 float64 的相对 RMS。

### 第一轮：照写的图（归一化用 Div）

| 模型 | 每次 execute | 其中 NPU | 线性层 tok/s | h2 误差 | k 误差 |
|---|---:|---:|---:|---:|---:|
| seg_m512_w4a16 | 174.8 ms | 167.2 | 52 | 8.01% | 15.81% |
| seg_m512_w8a8 | 88.3 ms | 83.1 | 104 | 22.37% | 19.88% |
| seg_m1024_fp16 | 289.9 ms | 270.5 | 63 | 0.07% | 0.07% |
| seg_m1024_w8a16 | 369.9 ms | 359.4 | 49 | 0.56% | 0.87% |
| seg_m1024_w4a16 | 347.5 ms | 337.0 | 53 | 8.01% | 15.81% |
| seg_m1024_w8a8 | 207.7 ms | 200.2 | 88 | 22.37% | 19.88% |
| seg_m1024_w4a8 | 218.2 ms | 211.5 | 84 | 24.19% | 26.45% |
| seg_m2048_w4a16 | 191.0 ms | 175.9 | 191 | 8.01% | 15.81% |
| seg_m2048_w8a8 | 107.4 ms | 98.9 | 341 | 22.37% | 19.88% |
| tiny_m16_fp16 | 1.57 ms | 0.81 | — | — | — |
| tiny_m16_w8a16（200 次） | 1.12 ms | 0.48 | — | — | — |

误差和 x86 上 HTP 模拟器的结果吻合：INT8 每行权重 + 16 位激活可用，4 bit 每行权重和 8 位静态激活都不能用。
详细 profile（逐算子周期）显示归一化的 Div 占了大头。但 HTP 会把矩阵乘和前后的算子融合，周期记到邻居名下
（m2048 的 o / gu / down / qkvg 都显示 0），所以逐算子计数只能看大概。

### 改图

| W8A16 | M = 1024 | M = 2048 | 误差 h2 / k / q |
|---|---:|---:|---|
| 照写（Div） | 369.3 ms | — | 0.56 / 0.87 / 1.29% |
| 归一化：每行一个倒数再乘 | 189.1 ms | 226.0 ms | 0.54 / 0.85 / 1.26% |
| 折叠（`make_models.py` 默认） | 66.9 ms | 165.2 ms | 0.57 / 0.87 / 1.29% |

折叠后其他量化方式：FP16（M = 1024）159.7 ms、0.07 / 0.06 / 0.07%；W8A8（M = 1024）44.5 ms、27 / 23 / 33%；
W4A16（M = 2048）140.9 ms、8.0 / 15.8 / 23%；FP16（M = 2048）312.8 ms。

折叠 W8A16 的块大小（两轮交替，第二轮在括号里）：

| M | 每次 execute | 其中 NPU | 线性层 tok/s |
|---|---:|---:|---:|
| 512 | 32.4 ms（32.4） | 28.0 | 283 |
| 768 | 47.1 ms（47.9） | 41.3 | 286 |
| 1024 | 66.5 ms（66.2） | 59.4 | 276 |
| 1536 | 108.9 ms（108.7） | 98.6 | 252 |
| 2048 | 165.2 ms（165.2） | 152.6 | 221 |

1024 以上块越大每行越慢，变慢的主要是归一化：M = 2048 的 h_yb 是 M = 1024 的 2.4 倍。原因可能是 [M, C] 的中间张量
放不进 8 MiB 的 VTCM，没有逐项核实。768 的 W8A8 是 27.4 ms，可作速度参考。

折叠 W8A16、M = 1024 的逐算子（[原始](data/y900-20261107-probe/npu-fold-detailed-seg_m1024_w8a16.txt)），合计 58.9 ms：
h_yb / h2_yb（o、down 输出的归一化）各约 11.5%，转回 FP32 输出 10.8%，down 7.2%，split_gu（含 gu 矩阵乘）6.3%，
SiLU 6.2%，o 4.8%，split_qkvg（含 qkvg）4.5%，其余是归一化和 slerp 的零碎。

### 和 GPU 同时、持续负载

| 场景 | 结果 |
|---|---|
| NPU 单独 / 旁边循环跑 gpu_probe | 每段 47.0 / 44.0 ms（NPU 上 41.3 / 40.7） |
| GPU 单独 / 旁边 NPU 连续 1500 次 | 矩阵乘 539 / 540 GFLOP/s，读带宽 66.0 / 55.7 GB/s |
| NPU 连续 10 × 250 次（231 s） | 每段 47.3–47.5 ms，NPU 41.9–43.4 °C、CPU 43–45 °C |

qnn-net-run 两次 execute 之间要读输入文件（每 250 次墙钟约 22 s，NPU 只占约一半），所以持续这一项不是满负荷。
第一轮里原定 3 分钟的持续测试（T4）被中途停掉，没有数据。

### App 能不能打开 CDSP（T8）

- `/vendor/etc/public.libraries.txt` 列了 `libcdsprpc.so`，App 能加载。
- `/dev/fastrpc-cdsp`：`crw-rw-r-- system system u:object_r:vendor_qdsp_device:s0`，其他用户可读。
- `run-as` 进开发版 App 的 uid 后打开被拒（Permission denied）。但 run-as 的 SELinux 域是 `runas_app`，不是 App 实际
  运行时的 `untrusted_app`，这一项还没有结论，要在 App 里实际调一次 QNN（方案的 N0，下一节）。

## N0：App 进程里的 NPU（11-07 夜）

开发版 APK 用 `-PxtgQnnLibs` 带上 QNN 的 HTP 库，以及改名成 lib*.so 的 qnn-net-run / qnn-profile-viewer（安装时解出到原生库目录）。
DEV 面板的「NPU 自检」或 `am start ... --ez npu_selftest true` 会从 App 进程跑 `files/npu/` 里的上下文二进制（`NpuProbe.kt`）。

- **第一次失败**（rc 11，"Device Creation failure"）：App 拉起的程序找不到 libcdsprpc.so。`/data/app` 下的程序进的是链接器的
  受限区段，看不到 /vendor/lib64；adb 用的 /data/local/tmp 是不受限的区段。LD_LIBRARY_PATH 加上 /vendor/lib64 后就通过了。
- **设备怎么来的**：App 进程自己 open /dev/fastrpc-cdsp 被 SELinux 拒（`avc: denied { open } ... untrusted_app ...
  vendor_qdsp_device`），vendor 策略只给 appdomain `ioctl read`。libcdsprpc.so 改从 `vendor.qti.hardware.dsp.IDspService`
  （dspservice）拿 fd，日志里有 `close device 8 thru HAL on domain 3`。
- **结果**（[原始](data/y900-20261107-probe/n0-selftest.txt)）：seg_m768_w8a16 两次，NetRun 46.0 / 45.7 ms、NPU 上 41.0 / 40.9 ms；
  tiny_m16_w8a16 1.0 / 0.4 ms，都和 adb 下一样。seg_m768 的 h2 / k / q 前 32 行和 adb 下逐字节相同。
- 清单加了 `<uses-native-library android:name="libcdsprpc.so">`：以后在 App 进程里直接加载 QNN（N1）时要用。

## S1：简单 agent 任务上的草稿接受数（11-07 夜）

- **任务**：`tools/agent_bench.py` 的 4 个会话，在本仓库源码上做小改动，起点 3K / 8K / 16K / 28K。每个会话最多 10 步，
  每次最多生成 8192 个 token。DSH 先不接。
- **生成**：在服务器上用 vLLM 0.33.0 跑（同一模型、2 循环、per_loop），`tools/draft_sim.py trace` 记下每个请求的提示和输出
  token，思考照样回传进历史。三组，共 87 个请求、6.2 万个 decode token。
- **模拟**：`draft_sim.py sim` 按 server/runtime.cpp 的规则：找上一次出现的 3-gram（没有就找 2-gram），取后面最多 k 个当草稿。
  因为"采样等于草稿才接受"，记下的序列给出的就是引擎会走的步数。核对：引擎实测的照抄请求是 49 步、草稿 387、接受 349，
  模拟得 49 / 392 / 350。
- 另外两种规则只是假设，引擎里没有："最长匹配"是 n 从 16 往下找；"3 条"是同时验证最近 3 处出现的续写。

每次前向出的 token（[原始](data/y900-20261107-probe/s1-sim-whatif.md)）：

| 组 | decode token（思考 / 正文 / 工具调用） | k = 8 | k = 16 | 最长匹配 k = 16 | 最长匹配 3 条 k = 16 |
|---|---|---:|---:|---:|---:|
| 默认温度（1.0 / top_p 0.95）、开思考 | 30,755（80% / 3% / 17%） | **1.90** | 2.00 | 2.02 | 2.09 |
| 　其中思考 | | 1.80 | 1.88 | 1.89 | 1.96 |
| 　其中工具调用 | | 2.86 | 3.17 | 3.23 | 3.38 |
| 默认温度、关思考 | 7,399（0 / 10% / 90%） | 2.09 | 2.21 | 2.25 | 2.33 |
| 贪心、开思考 | 23,955（65% / 4% / 31%） | 2.11 | 2.24 | 2.27 | 2.39 |
| 　其中工具调用 | | 4.35 | 5.56 | 5.97 | 6.46 |
| 每次验证的草稿行数 | | 7.9 | 15.6 | 15.7 | 31 |

- 思考的量最大，每次前向只有 1.8 个左右，草稿规则怎么改都上不去。工具调用（edit_file 照抄旧代码）高得多，贪心时能到 4–6。
- 按深度看没有单调规律，会话之间差别比深度之间大。每个会话的第一步思考都很长：2.8–6.9K 个 token，16K 那个会话
  写满 8192 也没调工具。
- 在服务器 GPU 上直接跑第一版 agent_bench（xtg-server，k = 8），停掉之前完成了几步：每个请求每次前向 1.4–3.9 个，
  和模拟是同一量级（[原始](data/y900-20261107-probe/s1-xtg-agent-bench-partial.log)）。那一版没回传思考，每一步都 cached 0，
  原因见 [speed-plan.md](../speed-plan.md) 的 P1。

## 数据文件

| 文件 | 内容 |
|---|---|
| `gpu_probe.txt` | GPU 摸底完整输出 |
| `engine-{cpu,gpu-linear,gpu}.txt` | 325 token generate，三个后端，含 GPU kernel 计时 |
| `engine-gpu-8k.txt` | 8000 token 提示，gpu 后端，per_loop |
| `engine-gpu-g1.txt`、`gpu_probe-gemvt.txt` | G1 第一步（k-major GEMV）之后的 325 token generate，和 GEMV 各写法的探针结果 |
| `spec-k{0,8}-server.txt`、`spec-k{0,8}-status.json` | 照抄请求，spec_k 0 / 8 的 kernel 计时和 `/xtg/status` |
| `npu-run1-*.txt` | 第一轮：环境、平台校验、App 域检查、每个模型的时间和内存 |
| `npu-run1-detailed-seg_m1024_w8a16.txt` | 第一轮照写图的逐算子 profile |
| `npu-norms.txt` | 照写 vs 倒数归一化（models = 照写，models2 = 倒数） |
| `npu-fold.txt` | 倒数归一化 vs 折叠（models3 = 折叠） |
| `npu-chunks.txt` | 折叠 W8A16 的块大小（models4 = 512 / 768 / 1536） |
| `npu-fold-detailed-seg_m1024_w8a16.txt` | 折叠图的逐算子 profile |
| `npu-concurrency-sustained.txt` | 和 GPU 同时跑、连续 4 分钟 |
| `n0-selftest.txt` | App 进程里的 NPU 自检 |
| `s1-sim.md`、`s1-sim-whatif.md` | 草稿接受数：引擎规则 k = 4 / 8 / 16，以及另外两种规则 |
| `s1-trace-{think,nothink,greedy}.log` | vLLM 生成的三组会话，每一步的深度、生成数、调用的工具 |
| `s1-xtg-agent-bench-partial.log` | 服务器上 xtg-server 跑第一版 agent_bench 完成的几步 |
