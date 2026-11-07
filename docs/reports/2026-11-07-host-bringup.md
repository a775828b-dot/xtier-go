# 2026-11-07 初版：服务器上的对照

[回到文档目录](../README.md)

初版的引擎、Dory 模型包和服务层在 x86 服务器上和 vLLM、纯 PyTorch 参考实现对照的结果。**没有在安卓设备上跑过**；手机上的
速度、温度、内存一个都没测。

## 环境

| 项 | 内容 |
|---|---|
| 机器 | 2× EPYC 9334（64 线程给引擎，通用内核，`-march=x86-64-v3`）+ RTX PRO 5000 72 GB（只给 vLLM 和参考实现） |
| 模型 | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4，HF 版本 64b4039a，model.safetensors sha256 ca08dec5… |
| 对照 | vLLM 0.33.0（含 PR #60149；torch 2.14.0+cu130；Marlin NVFP4 W4A16；BF16；`enforce_eager`）；`reference/dory_ref.py`（FP32，GPU） |
| 引擎配置 | 2 次循环、per_loop、prefill 分块 512（除非另写）；KV 写明 BF16 / FP8 |

## 结果

| 项 | 方法 | 结果 |
|---|---|---|
| 分词器 | `tests/tokenizer_check.py`：内置边界用例 + 4 个文件（模型 README、XTier 交接文档、Python 源码、模板）切片 | 203/203 与 tokenizers 逐 token 相同 |
| 模板 | `tests/template_check.py`：10 个用例 | 10/10 与 transformers 同款 jinja2 逐字符相同 |
| 前向对 FP32 参考 | 317 token 的带工具提示，整段 | logits 最大差 0.00041、平均 0.000015；argmax 100% |
| 滑窗环形缓冲 | 2600 token，分块 200（窗口 1024 回绕多次），BF16 KV | 对整段参考：logprob 平均差 0.0027、argmax 99.96%，NLL 2.8314 对 2.8317 |
| 解码路径 | `tests/decode_check.py`：贪心 48 token，再整段重算 | 48/48 是重算的 argmax |
| 对 vLLM（英文 512） | `reference/vllm_logprobs.py` 对 `xtg-cli forward` | logprob 平均差 0.017、argmax 98.4%，NLL 1.7443 对 1.7451 |
| 对 vLLM（中文 512） | 同上 | 平均差 0.020、argmax 97.5%，NLL 3.0066 对 3.0068 |
| 对 vLLM（带工具提示 317） | 同上 | 平均差 0.10、最大 6.78（第 103 位）、argmax 97.2%，NLL 4.157 对 4.131 |
| 接口 | `tests/api_check.py` | 12/12：普通回答、流式思考 + 工具调用（`read_file {"path": "src/main.cpp"}`，finish_reason tool_calls）、第二轮命中 367/405、`/generate` 逐位 logprob、`/flush_cache` |
| 照抄 | XTier `bench/copy_test.py --chunks 3` | xtier-go 2/3，vLLM 2/3：同一段、同一处漏掉同一行 |
| PPL | XTier `bench/ppl_eval.py --segments 2 --tokens 2048`（docs / agent / zh 三个语料） | 见下表 |
| aarch64 | QEMU 下 `kernels_test`、`parsers_test` | 通过，NEON 的 NVFP4 GEMV 与通用内核一致 |

PPL（每个语料 2 段 × 2048 token，冷 prefill、分块 512）：

| KV | docs | agent | zh | 全部 |
|---|---:|---:|---:|---:|
| BF16 | 1.69263 | 1.81022 | 2.80670 | 2.10318 |
| FP8（默认） | 1.69216 | 1.81353 | 2.81072 | 2.10547（+0.11%） |

## 带工具提示的那个大差：注意力很尖

逐层对比（`reference/vllm_layers.py` 在 vLLM 进程里挂钩子取每个子层的输出，对 `reference/dory_ref.py --bf16 --dump`）：

- 整体的相对误差从第 0 层 0.33% 平滑增长到最后约 7%，没有哪一层突变；
- 第 103 位在循环段第二轮的第 53–54 层开始放大（3.3% → 12.6%），到输出段第一层（56，滑窗注意力）跳到 65%；
- 把 vLLM 自己的第 55 层输出喂进参考实现的第 56 层，结果和 vLLM 的第 56 层只差 0.45%（这一位 0.46%）：**每层的算法和 vLLM
  一致**；
- 第 56 层在这一位的注意力分数：前两名 283.08 对 282.80、289.78 对 286.44……（nGPT 的 q 缩放让分数到几百）。0.28 的间隔
  比 BF16 舍入带来的分数误差还小，选哪个 key 就看舍入。

结论：差异是数值放大，不是实现错误。引擎的激活一律 FP32，比 vLLM 的 BF16 更接近精确值；这个位置 vLLM 碰巧更好
（`<required>` 后接 `["`：vLLM −0.11，我们 −6.9），但整段 NLL 只差 0.6%，自然文本上没有差别。

## KV 模式和精度（512 token，分块 64，对 vLLM）

| 配置 | 英文：平均差 / NLL | 中文：平均差 / NLL |
|---|---|---|
| 整段（不分块） | 0.0170 / 1.7443 | 0.0201 / 3.0066 |
| 分块 64，BF16 KV | 0.0173 / 1.7443 | 0.0202 / 3.0064 |
| 分块 64，FP8 KV | 0.0385 / 1.7481 | 0.0421 / 3.0097 |
| 分块 64，last_loop | 0.0402 / 1.7424 | 0.0650 / 3.0006 |

vLLM 那边是 per_loop、BF16、分块 8192（整段）。分块走缓存的路径结果不变；FP8 让偏差翻倍、NLL 高 0.1–0.2%；last_loop
和 per_loop 语义不同，偏差更大，NLL 没变差。用户定 KV 用 FP8（引擎和卡片默认）。

## 速度（只作参考，和手机无关）

x86 通用内核、64 线程：prefill 50–57 tok/s，decode 12.5–15.6 tok/s（2 次循环，几百到一千 token 的上下文）。同一机器上 vLLM
0.33.0（RTX PRO 5000，eager 模式）decode 约 11.6 tok/s。

## 过程中修掉的问题

- `/generate` 只生成 1 个 token 时，中文语料的 token 可能是半个汉字，返回 JSON 时 nlohmann 抛异常、httplib 返回 500；
  现在序列化时把不完整的 UTF-8 换成 U+FFFD。
- 内核测试的误差判据改成按输出的均方根归一化（aarch64 上编译器的求和顺序不同，接近 0 的输出会把相对误差放大）。

## 没验证的

安卓上的一切：App 界面、前台服务、下载、内存估算的灰掉、真机速度和温度、DSH Go X 的真实任务。NEON 只有 NVFP4 GEMV，
prefill 和注意力还是通用代码。

复现：命令都在 [validation.md](../validation.md)；提示词由 `reference/make_prompts.py` 生成（模型 README 的英文、一份中文
文档、带一个 `read_file` 工具的对话），长提示是同一份中文文档的 2600 token。
