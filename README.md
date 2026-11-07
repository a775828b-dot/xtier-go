# XTier Go

[English](docs/README.en.md) · [文档目录](docs/README.md) · [架构](docs/architecture.md) · [计划](docs/PLAN.md)

**GPU / NPU（开发中）：** 整步驻留 GPU 的路径在 Y900 上 prefill 31.9、decode 6.0 tok/s（CPU 7.7 / 4.4，325 token 提示，
输出一致），启动台默认仍是 CPU。目标（按实测调整）prefill 4K–8K 深度 250–300 / 32K 深度 120–180、decode 20–25 / 15–20 tok/s，路线、
NPU 摸底和分步计划见[速度方案](docs/speed-plan.md)。

XTier Go 是 [XTier](https://github.com/a775828b-dot/xtier) 给安卓手机和平板的一支：装在手机上的 App，打开是启动台，选模型和
启动选项后在本机起推理服务，其他 App（比如 DSH Go X）经 `http://127.0.0.1:18400/v1` 调用 OpenAI 兼容接口。推理核心是自己的
C++ 引擎（手机上不跑 SGLang / vLLM），模型照 vLLM main 的实现移植，在服务器上用 vLLM 对照验收。

**状态：早期开发，已在 Y900 上跑通 CPU 短请求。** App 安装、NEON/解析器、文件校验、长请求停止/重启以及现有 HTTP 接口验收通过；
模型经临时 USB 本地通道导入并校验，当前网络下独立镜像下载超时。DSH 客户端、长上下文、持续后台和 GPU 尚未验收。
服务器对照见下表，真机记录见 [Y900 首轮验证](docs/reports/2026-11-07-y900-bringup.md)。

- **语义照 vLLM**：Dory 的前向照 vLLM main（PR #60149）逐个算子移植；加载时核对 config.json 和分词规则，不认识就拒绝启动。
- **权重原样用**：NVFP4（E2M1 + E4M3 块缩放）直接从 safetensors mmap 出来算，不重新量化；E2M1 的 8 个幅值乘 2 都是整数，
  ARM 上一条 TBL 查表。
- **KV 用 FP8**：每个位置每个头一个缩放，按地址空间预留、写到才占内存；上下文最大 256K。
- **接口兼容**：OpenAI 兼容（流式、工具调用、思考内容、缓存命中上报），外加 SGLang 的 `/generate`、`/flush_cache`、
  `/metrics`，XTier 的测试脚本原样能测。

## 结构

```
engine/        引擎核心：线程池、safetensors mmap、内核（通用 + NEON）、KV 缓存（线性 / 环形，FP8 / BF16）、采样、模型包接口
packs/dory/    Dory 模型包：输入 / 循环 / 输出三段、per_loop / last_loop、分块 prefill、滑窗快照
server/        服务层：HTTP、OpenAI 兼容 + SGLang 子集、分词器、模板（minja）、思考 / 工具调用流式解析、前缀缓存、C API
app/           安卓 App（Kotlin、Jetpack Compose）：启动台、前台服务（独立进程）、模型文件下载与校验、JNI
configs/       每个「模型 × 设备」一张启动台卡片（card.json，和 XTier 的 console.py 同一套字段）
reference/     服务器上的对照：纯 PyTorch 参考实现、vLLM 的逐位 logprob 与逐层输出、transformers 同款模板渲染
tests/         内核、解析器、卡片（不要模型）；分词器、模板、前向、解码、逐层、接口（要模型）
tools/         xtg-server、xtg-cli、Unicode 表生成
third_party/   nlohmann/json、cpp-httplib、minja（单头文件，见 NOTICE）
```

## 快速上手（Linux）

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
build/kernels_test && build/parsers_test
build/xtg-server --model /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --threads 32 --opt context=131072
curl http://127.0.0.1:18400/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dory","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

启动选项（`--opt`）：`loops`（1–4，默认 2）、`kv_mode`（`last_loop` / `per_loop`）、`kv_dtype`（默认 `fp8`）、`context`
（最大 262144）、`prefill_chunk`、`snapshots`；模板默认值（`--kwarg`）：`medium_effort=true`、
`truncate_history_thinking=false`。Android NDK、aarch64 交叉编译、App 的构建见 [docs/build.md](docs/build.md)。

## 启动台（App）

打开就是启动台，平板上三栏（模型 · 启动参数 · 运行状态），手机上三个标签页：

- **模型**：每个模型一张卡片，写着端口和实测速度（还没在手机上测，现在是 `--`），选中的卡片显示文件是否齐全。
- **启动参数**：默认就是卡片里的配置，改过的项标出；放不下的上下文按可用内存灰掉并写明需要多少；没做的后端（GPU、NPU）
  灰掉写原因。模型文件点「下载模型」就行：直连 hf-mirror（域名用 223.5.5.5 解析，不经过代理），8 连接分段、断点续传、
  下完校验 sha256，放进 App 自己的目录；也可以 `adb push`。
- **运行状态**：加载阶段（检查文件 → 分词器与模板 → 映射权重 → 分配 KV → 预热 → 在线），在线后显示 decode / prefill 速度、
  上一个请求的缓存命中、内存、温度、电量和接口地址。

服务是独立进程里的前台服务：关掉启动台不影响它，一次只跑一个模型。

## 支持的模型

| 模型 | 模块 | 状态 |
|---|---|---|
| [Nemotron-Labs Dory 1.0 4B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) | `packs/dory` | 服务器对照通过；Y900 CPU 短请求和接口通过，长上下文未验收 |

## 支持的设备

| 设备 | 状态 |
|---|---|
| 联想 Y900（Snapdragon 8 Elite，SM8750P，16 GB，Android 16） | CPU 短请求、NEON 内核、解析器和启停回归通过；GPU、长时后台未验收 |
| x86-64 Linux | 开发和对照用（只有通用内核，不追求速度） |

## 上下文与内存（Dory，2 次循环，常驻估算）

| 上下文 | last_loop + FP8（默认） | per_loop + FP8 |
|---:|---:|---:|
| 32K | 3.18 GiB | 3.58 GiB |
| 128K | 4.89 GiB | 6.25 GiB |
| 256K | 7.18 GiB | 9.80 GiB |

按上下文用满算；KV 写到才占内存。细节和 decode 带宽上限见 [docs/dory.md](docs/dory.md)。

## 验证（服务器 x86，2026-11-07）

对照：vLLM 0.33.0（RTX PRO 5000，BF16）与纯 PyTorch 参考实现（FP32）。细节见[报告](docs/reports/2026-11-07-host-bringup.md)。

| 项 | 结果 |
|---|---|
| 分词器 | 203 段文本（中英文、代码、空白、特殊 token、多种文字）与 tokenizers 逐 token 相同 |
| 模板 | 10 个用例（工具、思考历史、工具结果、各模板变量）与 transformers 的 jinja2 渲染逐字符相同 |
| 前向对 FP32 参考 | logits 最大差 0.0004、argmax 100%；2600 token 分块 200（滑窗环形缓冲回绕）argmax 99.96% |
| 前向对 vLLM | 自然文本 NLL 1.7443 对 1.7451（英文）、3.0066 对 3.0068（中文）；带工具定义的提示个别位置不同（注意力很尖，逐层分析见报告） |
| 解码路径 | 逐 token 生成 48/48 是整段重算的 argmax |
| 接口 | 流式工具调用、多轮前缀复用（第二轮 405 个提示 token 命中 367）、`/generate`、`/flush_cache` 全过 |
| 照抄（XTier `copy_test.py`） | 2/3，与 vLLM 完全相同（同一段漏同一行，是模型自己的错） |
| PPL（XTier `ppl_eval.py`，3 个语料 × 2 × 2048） | FP8 KV 2.10547，BF16 KV 2.10318（FP8 高 0.11%） |

## 版本更新

- 2026-11-07：初版（0.0.1）：引擎、Dory 模型包、服务层、启动台 App、CI；服务器上对照 vLLM 验证，安卓未测。
  详见 [docs/reports/2026-11-07-host-bringup.md](docs/reports/2026-11-07-host-bringup.md)。

## 许可证

Apache-2.0（[LICENSE](LICENSE)）。第三方代码见 [NOTICE](NOTICE)；模型权重按 NVIDIA 的 OpenMDW 许可，不在仓库里。
