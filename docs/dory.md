# Dory 模型包

[回到文档目录](README.md)

[NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)（HF 版本
`64b4039a`）。代码在 `packs/dory/`，照 vLLM main 的 `vllm/model_executor/models/dory.py`（PR #60149）移植；纯 PyTorch 参考
实现在 `reference/dory_ref.py`。

## 结构

| 项 | 值 |
|---|---|
| 子层 | 72 个，注意力和 MLP 交替（每个子层只有其中一种）；输入 16 / 循环 40 / 输出 16（= 8 / 20 / 8 块） |
| 循环 | 循环段跑 1–4 次（模型卡推荐 2），每轮输入 = justnorm(state + input_h)，循环层权重共用 |
| 尺寸 | hidden 2560、MLP 7168、16 头 / 4 个 KV 头、head_dim 256、词表 131072，嵌入和 lm_head 不共享 |
| 注意力 | 每 4 层一层全局（9 层：输入 2、循环 5、输出 2），其余 27 层滑窗 1024（vLLM 的窗口：位置 t 看 t − 1023 .. t）；RoPE NeoX 式，θ 全局层 1e7、滑窗层 1e4 |
| 量化 | 所有 Linear 是 NVFP4（E2M1，每 16 个一个 E4M3 缩放 + 每个张量一个 FP32 缩放），W4A16；嵌入、lm_head 和各个缩放向量是 BF16 |
| 参数 | 39.75 亿：NVFP4 33.03 亿（打包 1.65 GB + 缩放 0.21 GB），BF16 6.72 亿；文件 2.98 GiB |

前向（vLLM 的写法，`justnorm(x) = x / sqrt(Σx² + 1e-6)`，`slerp(a, b, α) = justnorm(ŷa + |α·s|(ŷb − ŷa))`）：

```
h = embed(ids)
input 段:   每个子层 h = slerp(h, f(h), α)
input_h = h; state = input_h
loops 次:   h = justnorm(state + input_h); 循环段每个子层同上; state = justnorm(h)
output 段:  h = state; 每个子层同上
logits = (h · lm_headᵀ) · logit_scale / initializer_range
注意力 f:   q, k, v, g = 四个投影; RoPE(q, k); q = justnorm_per_head(q) · (sqk·√d/σ)²; k = justnorm_per_head(k)
            o = softmax(q·kᵀ/√d) v; o = o · sigmoid(g · (s_gate·√g0/σ)²); o = justnorm(o)（所有头一起）; o_proj
MLP f:      g, u = gate, up; m = silu(g · suv_gate·√H) · (u · suv_up·√H); m = justnorm(m); down_proj
```

## KV 模式

- **per_loop**（vLLM 默认）：循环段的注意力层每轮一份 KV（vLLM 用虚拟层号 `loop × 72 + layer`）。
- **last_loop**：循环段只存一份，后一轮覆盖前一轮；本次调用的 token 看本轮的 key，之前的 token 看最后一轮的。结果和分块
  大小有关（vLLM、llama.cpp PR 都这么说），也和 per_loop 略有不同。

全局层份数：per_loop 为 4 + 5 × 循环次数（2 次 = 14），last_loop 恒为 9。每层每个位置 K + V：BF16 4 KiB，FP8 2 KiB + 32 B 缩放。

## 内存（2 次循环，`xtg-cli estimate`）

常驻 = 权重（不含只读到用过的行的嵌入表）2.36 GiB + 全部 KV（按满上下文）+ 2 份滑窗快照 + 工作区。KV 按地址空间预留、写到
才占内存，这里按上下文用满算，启动台用它判断放不放得下。

| 上下文 | last_loop BF16 | last_loop FP8 | per_loop BF16 | per_loop FP8 |
|---:|---:|---:|---:|---:|
| 32K | 3.89 GiB | 3.18 GiB | 4.69 GiB | 3.58 GiB |
| 64K | 5.01 GiB | 3.75 GiB | 6.44 GiB | 4.47 GiB |
| 128K | 7.26 GiB | **4.89 GiB** | 9.94 GiB | 6.25 GiB |
| 256K | 11.76 GiB | 7.18 GiB | 16.94 GiB | 9.80 GiB |

Y900（15 GiB，平时可用约 7.9 GiB）上，128K 用 last_loop + FP8（卡片默认，4.89 GiB）或 per_loop + FP8（6.25 GiB）；
256K 只有 last_loop + FP8（7.18 GiB）勉强。BF16 两列只作对照，启动台不提供。

## 速度上限（按带宽估算，未实测）

decode 每个 token 要读一遍权重和 KV。按 60 GB/s 的有效带宽（8 Elite 理论约 85 GB/s，平板实际多少要测）：

| 循环 | 4K 上下文：每 token 读 | 上限 | 32K 上下文：每 token 读 | 上限 |
|---|---:|---:|---:|---:|
| 1 次 | 2.60 GiB | 21.5 tok/s | 3.59 GiB | 15.6 tok/s |
| 2 次 | 3.70 GiB | 15.1 tok/s | 5.23 GiB | 10.7 tok/s |
| 4 次 | 5.90 GiB | 9.5 tok/s | 8.52 GiB | 6.6 tok/s |

（BF16 KV、per_loop；last_loop 和 FP8 读得更少。）prefill 受算力限制，2 次循环每 token 约 10.3 GFLOP；128K 以上的注意力是
平方级，冷 prefill 在手机 CPU 上要按小时计，靠前缀缓存一轮一轮长上去。

## GPU 预算（Y900 的 Adreno 830，2026-11-07 实测）

全程持有 8 GiB 共享缓冲区时：复制读写合计 66.2 GB/s、Triad 67.5 GB/s、FP16 MAD 5.97 TFLOP/s、FP32 MAD 3.17 TFLOP/s、
简单 FP32 分块矩阵乘约 450 GFLOP/s；驱动报全局 7.5 GiB、单块上限 1 GiB。还没有 GPU 后端，下面是按这些数和模型的
字节 / 算力估的：

| 项 | 128K | 256K |
|---|---:|---:|
| 显存（last_loop + FP8：权重 2.36 + KV 与快照 + 工作区） | 约 5.1 GiB | 约 7.4 GiB |
| decode 上限（66 GB/s，4K / 32K / 128K 上下文） | 17.5 / 14.3 / 8.8 tok/s | — |
| prefill 估计（W4A16 FP16 矩阵乘 1.5–3 TFLOP/s，4K / 32K / 128K 处） | 125–250 / 80–160 / 35–75 tok/s | — |

按 8 GiB 的显存预算，128K 和 256K（last_loop + FP8）都放得下；单块 1 GiB 也够（最大的张量 lm_head 0.63 GiB，256K 时每个
全局层 KV 512 MiB）。前提是权重 CPU / GPU 共用一份（统一内存，mmap 的页直接做缓冲区）。decode 受带宽限制，CPU 和 GPU 共用
同一组内存，GPU 快不了多少；prefill 受算力限制，GPU 约快 3 倍，所以先做 GPU prefill（q·k 要 FP32 累加，见下一节）。

## 精度注意：注意力很尖

nGPT 的 q 缩放（`(sqk·√d/σ)²`，σ = 0.0177）让注意力分数到几百。服务器上对照时，一个带工具定义的提示里第 103 个位置的
滑窗层注意力分数是 283.08 对 282.80（只差 0.28）：BF16 和 FP32 的舍入就足以让它选另一个 key，后面的输出随之不同（vLLM
自己的第 55 层输出喂进参考实现，第 56 层只差 0.45%，所以每层的算法和 vLLM 一致，差异是放大出来的）。因此：

- 引擎激活一律 FP32；和 vLLM（BF16）比，自然文本平均差 0.02 nat，带工具定义的提示个别位置差得多；
- KV 定为 FP8（2026-11-07 用户定，引擎和卡片默认都是 FP8，BF16 只留给服务器上的对照）：key 的误差（约 2%）比这个间隔大，
  服务器上文本 NLL 比 BF16 高 0.1–0.2%；
- 要提 FP8 的精度，先试更细的缩放（每 32 个元素一个，而不是每个头一个），再看 agent 任务和照抄（[PLAN.md](PLAN.md)）。

## 分词器与模板

- tokenizer.json 是 Tekken 式的字节级 BPE（`ignore_merges`，1000 个特殊 token，`<think>`、`</think>`、`<tool_call>`、
  `</tool_call>`、`<|im_start|>`、`<|im_end|>` 都是单独的 token）。transformers 会提示这条预分词正则"不对"、建议
  `fix_mistral_regex=True`，vLLM 默认不改；xtier-go 和 vLLM 一样按 tokenizer.json 原样切。
- 模板变量：`enable_thinking`（默认开，提示以 `<think>\n` 结尾）、`medium_effort`（在最后一条用户消息后加
  `{reasoning effort: efficient}`）、`truncate_history_thinking`（默认开：用户发新消息时删掉之前各轮的思考）。模型卡说
  编码代理要传 `force_nonempty_content`，但这份模板里没有这个变量。
- 默认采样（generation_config.json）：温度 1.0、top_p 0.95、top_k 0；结束 token 2 和 11（`<|im_end|>`）。
- 工具调用格式（模板写死）：`<tool_call>\n<function=名>\n<parameter=参数>\n值\n</parameter>\n</function>\n</tool_call>`，
  值按工具 schema 转类型（vLLM 用 `qwen3_coder` 解析器，思考用 `nemotron_v3`）。
