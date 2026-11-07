# 版本更新

[回到文档目录](README.md)

- 2026-11-07：初版（0.0.1）：引擎核心（通用内核、NVFP4 GEMV 的 NEON 版、FP8 / BF16 KV、环形滑窗）、Dory 模型包、服务层
  （OpenAI 兼容 + SGLang 子集、分词器、模板、流式解析、前缀缓存与滑窗快照）、启动台 App 骨架、CI。服务器上对照 vLLM
  0.33.0 验证，安卓未测。详见 [reports/2026-11-07-host-bringup.md](reports/2026-11-07-host-bringup.md)。
