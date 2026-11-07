# 文档目录

[回到 README](../README.md) · [English](README.en.md)

| 文档 | 内容 |
|---|---|
| [architecture.md](architecture.md) | 架构：为什么手机上不跑 SGLang / vLLM，各层职责，从 XTier 继承的做法，设备后端路线 |
| [dory.md](dory.md) | Dory 模型包：结构与前向公式、KV 模式、内存表、decode 带宽上限、注意力很尖带来的精度问题、分词器与模板 |
| [speed-plan.md](speed-plan.md) | **速度方案**：DSH 32K 下 prefill / decode 的目标（11-08 按实测调整）、物理账、真机实测摘要、NPU + GPU 的分步计划与门槛、预计 |
| [gpu-adreno.md](gpu-adreno.md) | Adreno 830 GPU 加速方案：实测、prefill / decode 各项方法与估计收益、顺序、第一个原型（真机前写的，以 speed-plan.md 为准） |
| [api.md](api.md) | 本机接口：OpenAI 兼容对话、SGLang 子集、状态接口、前缀缓存 |
| [build.md](build.md) | 构建：Linux、aarch64 + QEMU、Android 命令行工具（adb shell）、App |
| [validation.md](validation.md) | 验收：每项检查的命令和通过条件，服务器上的参考环境和坑 |
| [PLAN.md](PLAN.md) | 工作计划：阶段、状态、已知问题 |
| [changelog.md](changelog.md) | 版本更新 |
| [reports/2026-11-07-host-bringup.md](reports/2026-11-07-host-bringup.md) | 初版在服务器上的对照报告 |
| [reports/2026-11-07-y900-bringup.md](reports/2026-11-07-y900-bringup.md) | Y900 首轮验证：App、CPU 短请求、接口 |
| [reports/2026-11-07-gpu-review-snapshot.md](reports/2026-11-07-gpu-review-snapshot.md) | GPU 线性层原型的审阅快照 |
| [reports/2026-11-07-y900-npu-gpu-probe.md](reports/2026-11-07-y900-npu-gpu-probe.md) | Y900 真机摸底：GPU 驻留路径、内核剖析、投机解码、NPU（图的写法、量化、块大小、并发）、App 里的 NPU、agent 任务的草稿接受数 |
