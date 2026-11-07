# 版本更新

[回到文档目录](README.md)

- 2026-11-07 晚（开发中，未发版）：启动台照 XTier 重做，加 `.dev` 并排开发版；整步驻留 GPU（`backend=gpu`）、FP16 分块
  矩阵乘、小 M GEMV、decode 注意力 split-K、按内核计时；提示查找投机解码（`spec_k`）；GPU / NPU 摸底工具（`tests/gpu_probe`、
  `tools/npu_probe`）。Y900 上 GPU 驻留 prefill 31.9 / decode 6.0 tok/s，速度方案见 [speed-plan.md](speed-plan.md)。
- 2026-11-07：初版（0.0.1）：引擎核心（通用内核、NVFP4 GEMV 的 NEON 版、FP8 / BF16 KV、环形滑窗）、Dory 模型包、服务层
  （OpenAI 兼容 + SGLang 子集、分词器、模板、流式解析、前缀缓存与滑窗快照）、启动台 App 骨架、CI。服务器上对照 vLLM
  0.33.0 验证，安卓未测。详见 [reports/2026-11-07-host-bringup.md](reports/2026-11-07-host-bringup.md)。
