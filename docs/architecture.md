# XTier Go 架构

[回到文档目录](README.md)

XTier Go 是装在安卓手机 / 平板上的推理 App：打开是启动台，选模型和启动选项后在本机起推理服务，其他 App（DSH Go X）
经 `http://127.0.0.1:18400/v1` 调用。它和 [XTier](https://github.com/a775828b-dot/xtier) 共用一套做法（固定上游语义、
加载自检、默认值 = 实测配置、照抄 / PPL / agent 测速验收），但推理核心不同：XTier 是 SGLang 的插件，XTier Go 是自己的
C++ 引擎。

## 为什么不在手机上跑 SGLang / vLLM

| 项 | 事实（2026-11） |
|---|---|
| Dory 的支持 | vLLM main 10-25 合入（PR #60149，75fc05a，0.33.0 起带）；SGLang main 没有；llama.cpp 是草稿 PR #30306 |
| ARM 支持 | SGLang / vLLM 的 ARM CPU 路径都以 BF16 为主，没有 NVFP4 |
| 安卓上跑 Python 版 | 要在 App 里带 glibc 用户空间（proot），targetSdk 只能 28，只能侧载 |
| 能用的计算单元 | 那样只剩 CPU：PyTorch 没有 Adreno / Hexagon 后端，高通的 OpenCL / QNN 库是 bionic 的 |
| 内存（2 次循环、32K） | BF16 权重 7.40 GiB，平板可用约 7.9 GiB，放不下 |

所以"插件"在 XTier Go 里落在两处：**模型包**（`packs/<模型>/`）和**设备后端**（CPU 先做，Adreno / Hexagon 以后）。
vLLM 留在服务器上当对照（`reference/`）。

## 分层

```
Android App（Kotlin，Jetpack Compose）
  启动台：模型卡片 · 启动参数 · 运行状态（平板三栏，手机三个标签页）
  模型文件：直连下载（域名经 223.5.5.5 解析，不走系统 DNS / 代理；8 连接分段、断点续传）· sha256 校验 · adb push 也行
      │ startForegroundService(启动 JSON)            ▲ http://127.0.0.1:18400/xtg/status（每秒）
      ▼                                             │
前台服务 EngineService（独立进程 :engine，specialUse，唤醒锁）
  └─ libxtiergo.so（JNI → server/xtg_capi.h）
       服务层 server/   HTTP（cpp-httplib）· OpenAI 兼容 + SGLang 子集 · 模板（minja）· 分词器
                        · 思考 / 工具调用流式解析 · 请求队列 · 前缀缓存（回退 / 滑窗快照）· 采样
       引擎核心 engine/ 线程池 · safetensors mmap · 内核（通用 + NEON）· KV 缓存（线性 / 环形，BF16 / FP8）
       模型包 packs/    dory/（I / R / O 三段、循环、per_loop / last_loop）
          ▲ OpenAI 兼容接口
     DSH Go X（安卓 App，或 proot Ubuntu 里的 DSH：proot 和安卓共用回环）
```

开发服务器上：`xtg-server` / `xtg-cli` 同一套源码在 x86 Linux 上跑，`reference/` 的 vLLM 与纯 PyTorch 参考实现
出对照结果，XTier 的 `bench/` 脚本直接测 `xtg-server`。

## 各层

### 引擎核心 `engine/`

- **与模型无关**：线程池（自旋后休眠，动态分块：手机大核小核频率不同）、safetensors mmap（不复制、不转换）、内核、KV 缓存、
  采样、模型包注册（`xtg/model.h`：`Model` / `Session` 接口）。
- **内核**（`xtg/kernels.h`）：激活一律 FP32，权重保持检查点格式按需解码。通用版本（`kernels.cpp`）定义结果、任何 CPU
  都能跑；aarch64 加 NEON 版本（`kernels_neon.cpp`，现在只有 NVFP4 GEMV），按运行时 CPU 特性选。
- **KV 缓存**（`xtg/kv.h`）：全局层线性缓存、滑窗层 1024 的环形缓冲；BF16 或 FP8（每个位置每个头一个 FP32 缩放）。
  按地址空间预留（`MAP_NORESERVE`），写到才占内存，所以 256K 容量不用一开始就占满。注意力内核读缓存里的旧位置和本次调用
  自己的新 key（还没写进缓存），短查询把 key 范围切给多个线程（flash-decoding）。
- **会话**：前缀复用要能回退。滑窗环形缓冲回退超过 1 个位置就丢了窗口里要用的旧 key，所以会话在每个提示词 prefill
  完时存一份滑窗快照（默认 2 份），回退不了时恢复到不晚于公共前缀的那份快照，再不行从零算。

### 模型包 `packs/dory/`

照 vLLM main 的 `dory.py` 逐个算子移植，细节见 [dory.md](dory.md)。加载时核对 config.json（model_type、
W4A16_NVFP4、层数分组、layer_types、RoPE），不认识就拒绝启动，不悄悄退化。启动选项：`loops`、`kv_mode`、`kv_dtype`、
`context`、`prefill_chunk`、`snapshots`。新模型 = `packs/` 下一个目录 + `packs/packs.cpp` 一行 + `configs/` 一张卡片。

### 服务层 `server/`

- **分词器**（`tokenizer.cpp`）：Hugging Face tokenizer.json 的字节级 BPE；加词表先按最长匹配，其余按预分词正则切块再合并。
  正则不解释执行，而是按 Tekken 的那条模式手写匹配（Unicode 类别表由 `tools/gen_unicode.py` 生成），tokenizer.json
  里的模式不同就拒绝加载。
- **模板**（`chat.cpp`）：minja 渲染 chat_template.jinja，照 vLLM 先把历史里工具调用的 arguments 从 JSON 字符串转成对象，
  不加 polyfill。
- **解析**（`parsers.cpp`）：`<think>`、`</think>`、`<tool_call>`、`</tool_call>` 都是单独的特殊 token，按 id 切状态；
  调用内部按 qwen3_coder 的 XML 解析，字符串参数边生成边流式输出（长文件内容不会憋到调用结束），其他类型按工具的 JSON
  schema 在参数结束时转换。写在思考里的调用照样算，结尾没闭合的调用保留。
- **运行时**（`runtime.cpp`）：后台加载（阶段：检查文件 → 分词器与模板 → 映射权重 → 分配 KV → 预热 → 在线），单个引擎线程
  串行处理请求（队列上限 8），一个会话的 KV 留给下一个请求复用。
- **HTTP**（`http.cpp`）：见 [api.md](api.md)。只听 127.0.0.1；同机任何 App 都能连回环，所以可选 API key。

### App `app/`

- **启动台**：原生 Compose 界面，字段和 XTier 控制台的卡片一样（`configs/<模型>-<设备>/card.json`：选项白名单、默认 =
  实测配置、改过的项标出、`off_when` 变灰、`unavailable` 带原因、notes）。不同的是选项分三类去向：`option`（模型包的启动
  选项）、`kwarg`（chat_template_kwargs 的默认值，例如思考力度）、`launch`（线程数等）。
- **内存检查**：App 用 `xtg_estimate`（不加载权重）对每个上下文 / KV 选项估算常驻内存，超过「可用内存 + 正在运行的服务 −
  1.5 GiB」的选项灰掉并写明需要多少。
- **服务**：`EngineService` 在独立进程 `:engine` 里，原生崩溃不带走界面；关掉启动台不影响服务；一次一个模型。
- **模型文件**：`Android/data/io.github.a775828b.xtiergo/files/models/<目录>/`。下载在前台服务 `DownloadService` 里跑：域名用阿里 DNS 的 HTTPS 接口（备用腾讯 HTTPDNS）解析，直连真实地址，不用系统 DNS（家里的 DNS 可能给代理的 fake-ip）；大文件 8 连接分段、断点续传，
  带 sha256 的文件可校验。

## 从 XTier 继承的做法

| XTier | XTier Go |
|---|---|
| 不改 SGLang，固定测试过的提交 | 不改 vLLM 的 Dory 语义；对照用 vLLM 0.33.0（含 PR #60149），卡片里记 HF 模型仓库的版本 |
| 加载后自检接管，没接管就退出 | 加载时核对 config.json 和分词器的预分词规则，不认识就拒绝 |
| 核心与模型无关，模型专用的进 `packs/` | 同 |
| `configs/<模型×硬件>/console.py` | `configs/<模型×设备>/card.json`（同一套字段，`build()` 改成声明式去向） |
| 默认 = 实测配置，改过的标出 | 同（现在还没有安卓实测，卡片测速写 `--`） |
| 一次一个模型，服务比控制台活得久 | 同（前台服务独立进程） |
| `mem_guard`、报在线前预热 | 内存估算灰掉放不下的选项；报在线前预热（读一遍权重页） |
| 照抄、PPL、agent_bench | 接口兼容，脚本原样用（[validation.md](validation.md)） |

## 设备后端路线

| 后端 | 用途 | 状态 |
|---|---|---|
| CPU 通用内核 | 正确性基准，任何 CPU | 完成，服务器上对照 vLLM |
| CPU NEON | decode + prefill | NVFP4 GEMV 完成（QEMU 下与通用版一致）；GEMM（i8mm）、注意力、范数待做 |
| Adreno（OpenCL） | prefill，以后 decode | 未开始；厂商 libOpenCL.so 要在 `/vendor/etc/public.libraries.txt` 里对 App 开放 |
| Hexagon（QNN） | prefill | 未开始；静态图，NPU 要重新量化，精度另验 |

启动台的「计算后端」选项里，没做的显示灰色和原因。
