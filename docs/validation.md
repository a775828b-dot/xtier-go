# 验收

[回到文档目录](README.md)

标准（沿用 XTier）：和参考实现比精度不差、XTier 的照抄 / PPL / agent 测试原样能跑；速度在安卓实测之后定门槛（建议 decode
达到实测带宽上限的 70%）。每一步改动先在服务器上过下面的检查，再上手机。

## 不需要模型

| 检查 | 命令 | 内容 |
|---|---|---|
| 内核 | `build/kernels_test`（aarch64：QEMU 下跑同一个） | E4M3 / BF16、NVFP4 矩阵乘对 double 参考、注意力对朴素 softmax（全局 / 滑窗 / 缓存 + 新 key / 切分 key，BF16 / FP8 缓存）、环形缓冲、采样、线程池；aarch64 上 NEON 对通用内核 |
| 解析器 | `build/parsers_test` | 思考 / 正文、带类型的调用参数（任意位置切开）、写在思考里的调用、没闭合的调用、长字符串流式、两个调用、无工具、不是调用、无参数 |
| 卡片 | `python tests/card_check.py` | 默认值在选项里、`off_when` / `notes` / `unavailable` 的键和值存在、kwarg 选项映射全 |

## 需要模型（服务器）

参考环境：vLLM 0.33.0（含 Dory）+ transformers 5.20 + tokenizers 的 venv。transformers 不认识 Dory 的配置，所以模板用
`reference/hf_chat.py`（transformers 同款 jinja2 环境）渲染、分词用 tokenizers。

| 检查 | 命令 | 通过条件 |
|---|---|---|
| 分词器 | `tests/tokenizer_check.py --cli build/xtg-cli --model M --files ...` | 和 tokenizers 逐 token 相同 |
| 模板 | `xtg-cli render` 对 `reference/hf_chat.py` | 逐字符相同（tools、思考、多轮、工具结果） |
| 前向 | `reference/make_prompts.py` 出 ids；`xtg-cli forward` 对 `reference/dory_ref.py`（FP32） | logits 最大差 < 1e-3、argmax 一致（分块、滑窗跨 1024 也要） |
| 对 vLLM | `reference/vllm_logprobs.py` 出 vLLM 的逐位 logprob，`tests/forward_check.py` 比 | 自然文本 NLL 差 < 0.5%；结构化提示的个别位置会因为注意力太尖而不同（[dory.md](dory.md)） |
| 逐层 | `reference/vllm_layers.py` + `reference/dory_ref.py --dump` + `tests/layers_check.py` | 定位用：每个子层对 vLLM 的相对误差，不应在某层突变 |
| 解码 | `tests/decode_check.py --cli build/xtg-cli --model M --ids P` | 逐 token 生成的每一步都是整段重算的 argmax |
| 接口 | `tests/api_check.py`（服务开着） | 普通回答、流式工具调用、多轮前缀复用、`/generate`、`/flush_cache` |
| 照抄 | XTier `bench/copy_test.py --port 18400 --model dory` | 和 vLLM 同样的结果（Dory 自己会漏行，见报告） |
| PPL | XTier `bench/ppl_eval.py --port 18400 --tokenizer <只有分词器文件的目录>` | 和参考在噪声内 |

注意：

- `ppl_eval.py` 用 `AutoTokenizer`，要指向只放 tokenizer.json / tokenizer_config.json / special_tokens_map.json 的目录
  （有 config.json 时 transformers 5.20 会因为不认识 Dory 报错）。
- vLLM 0.33.0 在这台机器上 FlashInfer 现编采样内核会失败（nvcc 13.5 和自带头文件不匹配），设
  `VLLM_USE_FLASHINFER_SAMPLER=0`；在 worker 里挂钩子要 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。
- ssh 里别用 `pgrep -f` / `pkill -f` 按命令行找服务（会匹配到 ssh 自己的 shell），用 pid 文件。

## 安卓（还没做）

真机第一步见 [PLAN.md](PLAN.md) 阶段 0：CPU 特性、前台服务所在 cpuset、内存带宽、持续负载温度，NVFP4 GEMV 原型到实测带宽
的 70%，然后 `adb shell` 里跑 `xtg-server`、电脑上 `adb forward` 后跑上面的接口、照抄、PPL。
