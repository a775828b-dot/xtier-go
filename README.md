# XTier Go

[English](docs/README.en.md) · [文档目录](docs/README.md) · [NPU 路线](docs/npu.md) · [架构](docs/architecture.md)

在安卓平板上本地跑一个 4B 推理模型，整步放在高通 NPU 上。XTier Go 是一个 App：打开是启动台，选好模型和档位，就在本机起一个
OpenAI 兼容的服务（`http://127.0.0.1:18400/v1`），DSH Go X 这类客户端直接连。模型是 NVIDIA 的
[Nemotron-Labs Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)（循环结构的 nGPT）。

> **现状（2026-11-08）：开发暂停。** 短对话（4K / 8K 档）可用：decode 12–16 tok/s，配提示查找 20–35；长上下文太慢（32K 档 decode 3.6 tok/s），DSH Go X 只能用极简模式。高通静态图每改一次都要重新导出（40–60 分钟），迭代太慢，NPU 路线到这里先停。

## 实测（联想 Y900，Snapdragon 8 Elite）

| 后端 · 档位 | prefill | decode | decode + 提示查找 |
|---|---:|---:|---|
| **NPU · 1 次循环 · 4K** | 342 tok/s | 15.5 tok/s | 20–35（随任务） |
| NPU · 1 次循环 · 8K | 162 | 12.5 | 约 15（DSH 实测） |
| NPU · 1 次循环 · 32K | 54 | 3.6 | — |
| NPU · 2 次循环 · 4K | 270–310 | 9.4 | 13 |
| GPU（Adreno 830，备用） | 42（325 token 提示） | 6.8 | — |
| CPU（NEON，备用） | 7.7 | 4.4 | — |

NPU 几行是平板上的实测（[报告](docs/reports/2026-11-08-y900-npu-dory.md)）。decode 受内存带宽限制：2 次循环每 token 要读约
3.5 GB；1 次循环少读约 1/3，代价是能力（官方 SWE-bench 59.5 → 50.3）。提示查找的效果看任务：照抄多的时候一步能验证 3–4 个
token，以思考为主时 1.2–1.5 个。**静态图每步都按整个档位算**，所以档位越大越慢；默认的档位列表是 4K → 8K → 32K，提示先在
小档位里算，放不下才搬到大的。DSH Go X 极简模式（基础提示 4.2K）实测：首轮 prefill 约 12 秒，之后每轮只算新增部分（1–2 秒），回答约 15 tok/s；一个简单的工具任务 3 分钟，其中约 1 分钟是 DSH 起会话标题的请求。标准模式基础提示 16K，要进 32K 档，太慢。

- NPU 这一路和高通官方的 Genie / AI Hub 同一套做法：服务器上用 ExecuTorch 的 Qualcomm 后端把模型编成静态图（16 位激活、
  4 位权重块、8 位 KV），手机上由 `libxtg_npu.so` 执行。分词、模板、采样、投机解码、前缀缓存和 HTTP 接口是自己的 C++ 引擎。
- GPU（Adreno）和 CPU 两条路还在，作为备用；它们直接读 NVFP4 权重，速度见[速度方案](docs/speed-plan.md)。

## 怎么做到的

- **两轮循环展开、各用一份权重。** Dory 的中间 40 个子层要跑两遍；HTP 编译器遇到"同一份权重被两个算子读"会把它变成每步
  重算的 DDR 张量，放不进单个 NPU 上下文。展开后多占约 1 GB 存储，每 token 读的字节不变。
- **切 4 片**：输入段 | 第 1 轮 | 第 2 轮 | 输出段，各是一个 QNN 上下文（V79 单个上下文约 3.5 GiB 封顶）。
- **KV 由插件管**：图只吃缓存、吐新行；全局注意力按位置存，滑窗层是固定 1088 槽的环，上下文再长也不涨；掩码每步按槽里的
  实际位置重建。回退、前缀缓存的快照都在插件里。
- **decode 图一次 8 行**：投机解码（提示查找）一步验证最多 8 个 token，耗时和算 1 个接近，因为 decode 卡在内存带宽上。
- **上下文档位**：静态图的 KV 长度固定，每个档位一个 `.pte`，每步都按整个档位算。一个会话可以列几个档位，每个 prefill 块
  都在放得下它的最小档位里算，放不下时插件把 KV 原样搬到下一个档位（不重算，搬一次 1–2 秒）。
- **注意力原地读缓存（4K、8K 档）**：一个 KV 头的 4 个 q 头叠在行上，和缓存直接相乘，缓存不复制，4K 每步 64 ms（常见写法
  72 ms）。32K 时 HTP 对这种大的分组乘法调度很差（680 ms），32K 档用常见写法（缓存和新行拼起来、K / V 按 q 头复制，280 ms）。
  三个档位用同一份校准，KV 编码相同，可以互相搬。HTP 编译器会把跨头广播的 RoPE 乘法编错（第一片之后 cos / sin 是图的
  输入），所以原地读的图里 RoPE 表先复制成 q / k 的完整形状。

细节、导出命令和每片的内存见 [docs/npu.md](docs/npu.md)。

## 用起来

1. 装 App（Actions 里有 debug APK；要用 NPU 得用带 QNN 库的开发版，见 [docs/build.md](docs/build.md)）。
2. 启动台选 **Dory 4B · NPU**：分词器等小文件可以直接下载；`.pte` 和同名的 `.kvmap.json` 目前没有下载源，在服务器上导出后
   `adb push` 到启动台显示的目录。
3. 选档位列表、投机解码，点启动。几秒后在线，接口 `http://127.0.0.1:18400/v1`，模型名 `dory`。
4. DSH Go X：设置 → 模型 → 添加模型提供商 → 自定义模型 API，地址填上面的接口、协议 OpenAI Chat Completions，API 密钥随便填
   一个（不能空），点「获取可用模型」；上下文窗口填最大档位（32768），最大输出 8192。DSH 标准模式的基础提示约 16K，
   在新会话里选「极简模式」只有约 4.2K，快得多。
5. 分屏或小窗时，App 只显示**速率表**：当前请求的 prefill / decode 速度、最近 32 个请求、内存和温度。

服务器上（Linux）也能编出同一个服务做对照：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
build/xtg-server --model /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --threads 32 --opt context=131072
```

## 结构

```
npu/           NPU 插件 libxtg_npu.so（C 接口；KV、掩码、回退、快照）和测试工具 npu_test
tools/npu_et/  NPU 导出：Dory 的静态图模型、权重转换、ExecuTorch 补丁、导出脚本、CPU / 设备上的对照
engine/        引擎核心：线程池、safetensors mmap、内核（通用 + NEON + OpenCL）、KV 缓存、采样、模型包接口
packs/dory/    Dory 模型包：CPU / GPU 前向，backend=npu 时转给插件
server/        服务层：HTTP、OpenAI 兼容 + SGLang 子集、分词器、模板、思考 / 工具调用流式解析、前缀缓存、投机解码
app/           安卓 App（Kotlin、Jetpack Compose）：启动台、速率表、前台服务、模型下载与校验
configs/       每个「模型 × 设备」一张启动台卡片
reference/     服务器上的对照：纯 PyTorch 参考实现、vLLM 逐位 logprob
tests/         内核、解析器、卡片（不要模型）；分词器、模板、前向、解码、接口（要模型）
```

## 精度

同一段 2048 token 的 agent 文本，逐位置的下一 token NLL：BF16 参考实现 1.104，NPU（2 次循环）1.093，持平。量化的损失按
CPU 模拟主要来自 4 位权重（argmax 和 BF16 一致 87%），8 位 KV 几乎没有影响。导出时 V 缓存和新 V 被量化成不同的编码，
插件写回缓存时按编码映射（`NAME.kvmap.json`）——不做这一步 NLL 会到 1.43。

服务器上 CPU 路径对 vLLM 0.33.0 的验收（分词、模板、logits、接口、照抄）见
[reports/2026-11-07-host-bringup.md](docs/reports/2026-11-07-host-bringup.md)。

## 版本更新

- 2026-11-08（0.0.2）：**整步跑在 Hexagon NPU 上**：ExecuTorch QNN 静态图导出（`tools/npu_et`）、NPU 插件（`npu/`，自己管
  KV、8 行验证图、上下文档位 4K / 8K / 32K、提示按块在小档位里先算、注意力原地读缓存）；启动台按 NPU 改、分屏速率表；长 prefill 保活；
  DSH Go X 接通。
- 2026-11-07：初版（0.0.1）：引擎、Dory 模型包、服务层、启动台 App、CI；GPU 整步驻留和提示查找投机解码。
  详见 [docs/changelog.md](docs/changelog.md)。

## 许可证

Apache-2.0（[LICENSE](LICENSE)）。第三方代码见 [NOTICE](NOTICE)；模型权重按 NVIDIA 的 OpenMDW 许可，不在仓库里；QNN / QAIRT
的库按高通的许可，不在仓库里。
