# 本机接口

[回到文档目录](README.md)

`xtg-server`（Linux / adb shell）和 App 里的服务是同一套代码（`server/http.cpp`），默认只听 `127.0.0.1:18400`。启动时可以设
API key（`--api-key`，或启动 JSON 的 `api_key`），设了之后 `/v1/*`、`/generate`、`/flush_cache` 要带
`Authorization: Bearer <key>`；`/health`、`/metrics`、`/xtg/status` 不需要。

| 接口 | 说明 | 谁用 |
|---|---|---|
| `GET /health` | 在线 200，加载中 503 | 启动台、脚本 |
| `GET /v1/models` | `[{id: "dory", max_model_len}]` | DSH Go X |
| `POST /v1/chat/completions` | OpenAI 兼容，见下 | DSH Go X、XTier `copy_test.py`、`agent_bench.py` |
| `POST /generate` | SGLang 的子集，见下 | XTier `ppl_eval.py`、`kl_compare.py` |
| `POST /flush_cache` | 丢掉缓存的前缀，下一个请求从零算 | `ppl_eval.py` |
| `GET /metrics` | Prometheus 文本：`xtg:requests_total`、`prompt_tokens_total`、`generation_tokens_total`、`cached_tokens_total`、`gen_throughput`、`prefill_throughput`、`rss_bytes` | 监控 |
| `GET /xtg/status` | 状态 JSON：`state`（loading / online / failed / stopped）、`phase`、`phase_label`、`progress`、`error`、`model`（启动选项与内存估算）、`telemetry` | 启动台（每秒） |

## `/v1/chat/completions`

请求字段：`messages`（必需；content 可以是字符串或 text 片段数组）、`tools`、`tool_choice`（`none` 时不解析调用）、`stream`、
`stream_options.include_usage`、`max_tokens` / `max_completion_tokens`（缺省 = 上下文剩下的）、`temperature`、`top_p`、`top_k`、
`min_p`（缺省取 generation_config.json：1.0 / 0.95 / 0 / 0）、`seed`、`stop`、`chat_template_kwargs`（盖在启动台给的默认值上，
Dory：`enable_thinking`、`medium_effort`、`truncate_history_thinking`）。`model` 字段不检查。

返回：

- 非流式：`choices[0].message` 有 `content`（只有工具调用时为 null）、`reasoning_content`、`tool_calls`
  （`arguments` 是 JSON 字符串）；`finish_reason` 是 `stop` / `length` / `tool_calls`；`usage` 带
  `prompt_tokens_details.cached_tokens`（这次复用了多少前缀）。
- 流式（SSE）：第一块 `delta: {role: assistant, content: ""}`；之后 `reasoning_content`、`content`、`tool_calls` 的增量
  （调用的第一块带 `id`、`type`、`function.name`，之后只有 `function.arguments` 的片段，按 `index` 拼起来就是完整 JSON）；
  最后一块 `delta: {}` 带 `finish_reason`；`include_usage` 时再来一块 `choices: []` + `usage`；`data: [DONE]`。
- 提示词以 `<think>\n` 结尾（`enable_thinking` 默认开）时，输出先算思考，`</think>` 之后是正文。

前缀缓存：每个请求和上一个请求的 token 比最长公共前缀，复用这一段的 KV。滑窗层是环形缓冲，回退超过 1 个位置时恢复到
不晚于公共前缀的快照（每个提示词 prefill 完存一份，默认留 2 份），否则从零算。agent 循环里模板会把上一轮的思考和调用原样
渲染回来，实测第二轮 405 个提示 token 命中 367 个（= 上一轮提示 + 生成）。

## `/generate`（SGLang 子集）

请求：`input_ids` 或 `text`（按特殊 token 解析）、`sampling_params`（`max_new_tokens` 缺省 128、`temperature`、`top_p`、
`top_k`、`min_p`、`stop`）、`return_logprob`、`logprob_start_len`、`top_logprobs_num`。`return_logprob` 时不复用前缀
（每个位置都要 logits）。

返回：`text`（原样字节，不完整的 UTF-8 换成 U+FFFD）、`output_ids`、`meta_info`：`finish_reason.type`、`prompt_tokens`、
`completion_tokens`、`cached_tokens`、`input_token_logprobs`（`[logprob, id, null]`，`logprob_start_len = 0` 时第一个是
`[null, id, null]`）、`input_top_logprobs`。
