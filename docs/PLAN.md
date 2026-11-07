# 工作计划

[回到文档目录](README.md)

## 目标

一个装在安卓手机 / 平板上的推理 App：启动台选模型和选项、本机起服务，DSH Go X 经 `127.0.0.1:18400/v1` 调用。首个模型
Dory 1.0 4B NVFP4，首台设备联想 Y900（SM8750P，15 GiB）。上下文要求 256K，至少 128K；KV 用 FP8。

## 原则

- 平板上不跑 SGLang / vLLM：自己的 C++ 引擎，模型照 vLLM main 移植，vLLM 在服务器上当对照（[architecture.md](architecture.md)）。
- 核心与模型无关，模型专用的进 `packs/`，设备后端可插拔。
- 默认值 = 实测配置；没实测的写明。
- 每步有门槛（[validation.md](validation.md)）；改解码路径先过照抄和逐 token 解码检查。

## 阶段

2026-11-07 Y900 首轮进展：App 安装、NEON/解析器、校验与启停回归、短请求 HTTP 验收已通过；
CPU 冷提示 prefill 6.66–7.19 tok/s、decode 3.68–4.38 tok/s（两组短请求，各测一次）。
模型通过临时 USB 通道导入并校验，直接镜像下载在当前网络超时。阶段 0 / 2b 只完成部分，
DSH 真客户端、10/30 分钟稳定性、长上下文与 GPU 不在此次通过范围。详见[记录](reports/2026-11-07-y900-bringup.md)。

2026-11-07 晚：整步驻留 GPU 跑通（prefill 31.9、decode 6.0 tok/s，输出与 CPU 一致），提示查找投机解码（`spec_k`）上线但
GPU 小 M 内核没复用权重、只快 19%；NPU 摸底：折叠后的 W8A16 段线性层折算 286 tok/s。速度目标（11-08 按实测调整为 prefill 250–300 / 120–180、decode 20–25 / 15–20，分别是 4K–8K 和 32K 深度）
（DSH 32K）的分步计划见 [speed-plan.md](speed-plan.md)，数据见[摸底报告](reports/2026-11-07-y900-npu-gpu-probe.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 引擎 + Dory + 服务（服务器上） | 通用内核、NVFP4、KV（线性 / 环形、FP8）、Dory 前向（两种 KV 模式、分块）、分词器、模板、解析器、HTTP、C API、命令行工具 | **完成**（2026-11-07，[报告](reports/2026-11-07-host-bringup.md)） |
| 1b App 骨架 | Compose 启动台、前台服务、JNI、模型文件下载 / 校验、内存估算 | 写完，CI 编译；**没在设备上跑过** |
| 0 真机摸底 | `/proc/cpuinfo`（i8mm / bf16 / sve）、前台服务的 cpuset、内存带宽（1 / 2 / 6 / 8 线程）、持续负载温度、`libOpenCL.so` 是否对 App 开放；`adb shell` 跑 `kernels_test`、`xtg-server` | GPU / NPU 部分完成（App 里 OpenCL 可用，[报告](reports/2026-11-07-y900-npu-gpu-probe.md)）；App 里的 NPU 已通过（经 dspservice）；cpuset、CPU 带宽、满负荷持续温度待做 |
| 2 CPU 速度 | NEON：NVFP4 GEMV 做到实测带宽 70%、GEMM（i8mm W4A8，A/B 精度后再定默认）、注意力（FP8 KV 直接解码）、范数 / RoPE；线程绑核 | 待做 |
| 2b 真机验收 | App 真机启动、DSH Go X 真实任务、息屏 / 后台 30 分钟、首请求不卡、内存不触发 LMK、10 分钟持续速度 | 待做 |
| 3 加速包 | FP8 KV 更细的缩放（每 32 个元素）、提示词查找投机解码（n-gram，验证 k 个 token 与 decode 一步读的权重一样多）、多个会话槽 | 投机解码已实现（`spec_k`，per_loop 精确），等 GPU 小 M 内核（[speed-plan.md](speed-plan.md) G2 / G3）；agent 任务上每次前向约 1.9 个（S1），要更多得做草稿头（S2）；其余待做 |
| 4 GPU / NPU | Adreno（OpenCL）整步驻留；Hexagon（QNN）做 prefill 的线性层，GPU 同时做注意力。按 [speed-plan.md](speed-plan.md) 的 N0 → G1–G5 / N1–N3 推进 | 进行中：GPU 驻留已跑通，NPU 摸底完成 |
| 5 更多模型 | 新模型 = `packs/` 一个目录 + `configs/` 一张卡片 | 待做 |

## 已知问题与限制

- x86 上只有通用内核，速度只够对照；手机上的速度一个都没测过，卡片测速写 `--`。
- `stop` 字符串命中后会停，但已经流出去的那部分不回收。
- 回退超过 1 个位置且没有可用快照时从零 prefill（`snapshots` 默认 2 份）。
- 一次只有一个会话：两个客户端交替请求会互相挤掉前缀缓存。
- `force_nonempty_content`（模型卡给编码代理的建议）在这份模板里不存在，没做处理。
