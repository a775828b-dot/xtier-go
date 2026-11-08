# 版本更新

[回到文档目录](README.md)

- 2026-11-08（0.0.2）：**整步跑在 Hexagon NPU 上**（`backend=npu`）：`tools/npu_et` 用 ExecuTorch v1.5.1 + QAIRT 2.37 把 Dory
  导出成静态图（两轮循环展开、切 4 片、16 位激活 / 4 位权重块 / 8 位 KV），`npu/libxtg_npu.so` 执行（KV 由插件管：全局按位置、
  滑窗环形；8 行 decode 图给投机解码验证；V 缓存按编码映射写回（`NAME.kvmap.json`，导出时生成）；上下文档位 4K / 8K / 32K，
  每个 prefill 块在放得下它的最小档位里算、放不下时把 KV 原样搬过去；注意力原地读缓存，RoPE 表复制成完整形状绕开 HTP
  跨头广播的编译错误）。Y900 4K 档：1 次循环 prefill 约 400、decode 14.4 tok/s
  （照抄多的任务配提示查找 51.5），2 次循环 280–290 / 9.4；1 次循环 8K 档 prefill 162 / decode 12.5，32K 档 54 / 3.6；插件的共享内存只放一份缓存（之前长档位 DSP 映射不下、
  每步拷贝）；前缀缓存在客户端不回传思考时也能复用（DSH 之前每轮从零算）。长上下文太慢，开发暂停。启动台按 NPU 改（档位列表、.pte 走 adb push、
  环形布局的内存估算），分屏 / 小窗只显示速率表；`/xtg/status` 报实时请求和最近 32 个请求；长 prefill 每 15 秒发空 delta 保活。
  DSH Go X 已接通。详见 [reports/2026-11-08-y900-npu-dory.md](reports/2026-11-08-y900-npu-dory.md)。

- 2026-11-07 晚（开发中，未发版）：启动台照 XTier 重做，加 `.dev` 并排开发版；整步驻留 GPU（`backend=gpu`）、FP16 分块
  矩阵乘、小 M GEMV、decode 注意力 split-K、按内核计时；提示查找投机解码（`spec_k`）；GPU / NPU 摸底工具（`tests/gpu_probe`、
  `tools/npu_probe`）；App 内 NPU 自检（`-PxtgQnnLibs`）；简单 agent 任务测速 `tools/agent_bench.py`、草稿模拟
  `tools/draft_sim.py`。Y900 上 GPU 驻留 prefill 31.9 / decode 6.0 tok/s，速度方案见 [speed-plan.md](speed-plan.md)。
- 2026-11-07：初版（0.0.1）：引擎核心（通用内核、NVFP4 GEMV 的 NEON 版、FP8 / BF16 KV、环形滑窗）、Dory 模型包、服务层
  （OpenAI 兼容 + SGLang 子集、分词器、模板、流式解析、前缀缓存与滑窗快照）、启动台 App 骨架、CI。服务器上对照 vLLM
  0.33.0 验证，安卓未测。详见 [reports/2026-11-07-host-bringup.md](reports/2026-11-07-host-bringup.md)。
