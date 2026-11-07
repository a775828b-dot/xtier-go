# Adreno 830 GPU 加速方案

[回到文档目录](README.md)

**状态：还没实现**，引擎现在只有 CPU 后端。下面的收益都是按 Y900 上的实测数（2026-11-07）和 Dory 的字节数 / 算力估的，
实现后以实测为准。

## 实测（Y900，Snapdragon 8 Elite，Adreno 830，Android 16，OpenCL 3.0，驱动 0800.72）

全程持有 8 GiB 共享缓冲区（32 × 256 MiB）时：

| 项 | 结果 |
|---|---:|
| 复制读写合计 / Triad | 66.2 / 67.5 GB/s |
| FP16 / FP32 MAD | 5.97 / 3.17 TFLOP/s |
| 简单 FP32 分块矩阵乘（64×64 块，每线程 4×4） | 444–455 GFLOP/s |
| 驱动报告 | 全局 7.5 GiB，单块上限 1 GiB |

分配 8 GiB 后系统还剩约 2.8 GiB 可用。显存预算按 8 GiB：128K（last_loop + FP8）约 5.1 GiB、256K 约 7.4 GiB，都放得下
（[dory.md](dory.md#gpu-预算y900-的-adreno-8302026-11-07-实测)）。

## 原则

- **prefill 是算力瓶颈**：2 次循环每 token 约 10.3 GFLOP（线性层），长上下文再加注意力（128K 处约 30 GFLOP）。GPU 的 FP16
  算力约是 CPU 的 3 倍，DSH 一上来上万 token 的提示主要靠它。**先做 GPU prefill。**
- **decode 是带宽瓶颈**：每 token 读权重 3.3 GiB（+ KV），66 GB/s 是 CPU 和 GPU 共用的，GPU 本身快不了多少。要快只能少读
  字节、少发 kernel，或者用 GPU 富余的算力做投机解码。
- **精度**：Dory 的注意力很尖（分数几百、前两名常只差零点几，[dory.md](dory.md#精度注意注意力很尖)），q·k 一律 FP32 累加；
  矩阵乘的累加也用 FP32（或 FP16 分段累加再并入 FP32）。

## Prefill

| 方法 | 估计收益 | 代价 / 风险 |
|---|---|---|
| **FP16 分块矩阵乘**，NVFP4 在寄存器里解成 FP16：E2M1 × E4M3 的乘积最多 6 位有效数字，FP16 精确表示；每个张量的全局缩放放到累加之后 | 短上下文 44 → 125–250 tok/s（FP16 峰值 5.97 TFLOP/s，简单 FP32 核只有约 450 GFLOP/s） | 累加精度要按上面的原则 |
| **权重加载时重排一次**：量化码和缩放分开连续存（SoA），按 work-group 的 tile 排；大矩阵走 image（纹理缓存）；subgroup 规约 | 上一项跑到 40–60% 峰值的前提（高通在 IWOCL 2025 讲的 Adreno 内核写法） | 转换一次，结果缓存在模型目录 |
| **分块注意力**（flash-attention 式）：直接读 FP8 KV（内核里解码），QK FP32 累加，PV FP16 | 128K 以上 prefill 的大头 | — |
| 每块 1024–2048 token | 矩阵乘的 M 够大 | 激活多占几十 MiB |
| int8 点积（`cl_qcom_dot_product8`，W4A8） | 可能再快一倍 | 激活量化有误差，要 A/B |

## Decode

| 方法 | 估计收益 | 代价 / 风险 |
|---|---|---|
| **合并内核**：q / k / v / g 一个 GEMV；RoPE + qk 归一化 + 缩放一个核；SwiGLU + 归一化并进下一个矩阵乘。nGPT 的 justnorm 是每行一个标量，W(x/‖x‖) = (Wx)/‖x‖，"先归一化再乘"可以改成"乘完再除"，省掉单独的归一化 | 每 token 本来 300–450 个 kernel（每个几十微秒），省 10–20% | 工作量中等 |
| **高通 recordable queue**（`cl_qcom_recordable_queues`）：kernel 序列录一次，每 token 只改参数重放，类似 CUDA 图 | 和上一项叠加，压掉 CPU 发 kernel 的开销 | 先用 `clinfo` 确认 830 的驱动有这个扩展（730 有） |
| **投机解码在 GPU 上几乎免费**：一次验证 8 个 token 读的权重和 decode 一步一样（3.3 GiB，约 77 ms），多出的约 82 GFLOP 计算 GPU 40–80 ms 算完；CPU 上要慢 2–4 倍。用提示词查找（n-gram），不需要草稿模型 | 编码代理（写代码、改文件）decode 估计 1.5–2.5 倍 | 思考文本的接受率会低，要实测 |
| **lm_head BF16 → FP8**（每行一个缩放） | 每 token 少读 0.31 GiB，decode +10%；NVFP4 约 +15% | 要验精度 |
| V 存 4 bit（K 保持 FP8） | 128K 时 KV 每 token 读 3.6 GiB、比权重还多，约 +13% | KV 定的是 FP8，只作可选实验；K 不能降 |

## 全局

- **权重只放一份**：用 `CL_MEM_ALLOC_HOST_PTR`（或 dmabuf）分配的缓冲区装权重，CPU 持久映射同一份（统一内存）。省 2.4 GiB，
  8 GiB 预算里给 KV 留空间。单块上限 1 GiB：按张量分块（最大的 lm_head 0.63 GiB，256K 时每个全局层 KV 512 MiB）。
- **频率**：`cl_qcom_perf_hint` 要高频；长 prefill 会热降频，用持续性能模式换稳定，测连续 10 分钟后的速度。
- CPU 和 GPU 同时做 prefill 理论 +25–30%（CPU i8mm 约 0.6–1 TFLOP/s），但两边抢同一份散热和内存，可能不划算，放最后。
- Hexagon NPU 做 prefill 潜力更大，但要静态图和重新量化，不在这份方案里。

## 叠起来（估算，4K 上下文）

| | 现在（CPU 估算） | GPU 基础版 | 加上合并 / recordable queue / lm_head FP8 | 再加投机解码（编码任务） |
|---|---:|---:|---:|---:|
| prefill | 50–85 tok/s | 125–250 | — | — |
| decode | 带宽上限 15 tok/s | 10–13 | 13–16 | 20–35 |

int8 点积能用的话 prefill 约 200–400 tok/s。长上下文时 prefill 和 decode 都按 [dory.md](dory.md) 的表往下掉。

## 顺序

1. **摸底与原型**（平板接上后）：`clinfo` 确认 `cl_qcom_recordable_queues`、`cl_qcom_dot_product8`、`cl_qcom_perf_hint`、
   subgroup、image 尺寸上限、`CL_MEM_ALLOC_HOST_PTR` 的零拷贝行为；写一个 NVFP4 → FP16 GEMM 原型（2560 × 7168，M = 512，
   和 `kernels_test` 一样对 double 参考校验），测出实际 TFLOP/s：prefill 能到多少由它定。原型用 CI 编成 `adb shell` 能跑的程序。
2. **GPU prefill**：矩阵乘 + 分块注意力 + 权重只放一份；decode 先留在 CPU（NEON）。
3. **decode**：合并内核 + recordable queue → lm_head FP8 → 提示词查找投机解码。
4. 有精度风险的（int8 点积、V 存 4 bit）最后，都要和现在的版本 A/B（PPL、照抄、对 vLLM 的逐位 logprob）。

后端接口按 [architecture.md](architecture.md) 的设备后端走：启动台「计算后端」选项的 GPU 现在是灰的，做完一段放开一段。

## 参考

- [IWOCL 2025：高通 Adreno OpenCL 后端的内核写法](https://www.iwocl.org/wp-content/uploads/iwocl-2025-hongqiang-wang-lamacpp-backend-update.pdf)
- [Qualcomm：Adreno 上的 OpenCL GPU 后端](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu)
- [cl_qcom_recordable_queues 说明](https://blog.csdn.net/weixin_38498942/article/details/129180255)
- [Adreno 730 的 OpenCL 扩展列表（含 recordable queues）](https://gist.github.com/woachk/9a0c542a946e4d2838564f104e0833d7)
