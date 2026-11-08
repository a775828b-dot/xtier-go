# NPU 路线（Hexagon，ExecuTorch QNN 静态图）

整步在高通 NPU 上跑：模型在服务器上用 ExecuTorch 的 Qualcomm 后端导出成 `.pte`（里面是 QNN 上下文二进制），手机上由
`libxtg_npu.so` 执行；引擎的 Dory 模型包用 `backend=npu` 经 dlopen 调它，分词、模板、采样、投机解码、前缀缓存和 HTTP 接口都还是
引擎自己的。做法和高通官方 Genie / AI Hub 那一套相同：16 位激活、4 位权重块、8 位 KV，prefill 一张多行图、decode 一张少行图。

## 组成

| 位置 | 内容 |
|---|---|
| `tools/npu_et/dory/` | 静态图模型 `model.py`（循环展开）、BF16 权重转换并折叠缩放 `convert_weights.py`、校准复用 `calib_cache.py` |
| `tools/npu_et/apply_dory_patch.py` | 给 ExecuTorch v1.5.1 的 Qualcomm LLM 示例注册 `dory-4b`：量化配方、按循环边界切片、KV / 掩码形状、校准复用 |
| `tools/npu_et/export_dory.sh` | 导出一个上下文档位 |
| `tools/npu_et/test_dory_ref.py` | CPU 上按插件的方式跑静态图（分块、环形缓存、掩码），对 `reference/dory_ref.py` |
| `tools/npu_et/compare_score.py` | 设备上逐位置的 NLL / argmax（`npu_test … score`）对参考实现 |
| `npu/xtg_npu.{h,cpp}` | 插件：C 接口；KV 缓存、两张掩码、回退、环形缓存快照都在这里，缓存放 RPC 共享内存，NPU 原地读 |
| `npu/npu_test.cpp` | 只用插件的测速、回退一致性、批量验证一致性和逐位置打分 |
| `packs/dory/npu.cpp` | 引擎侧：dlopen 插件、按块报进度和取消、按块选档位、回退、快照 |

## 图

- **量化**：从 **BF16 版**（`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`）量化，权重 4 位整数、每 16 个一块（和 Dory 训练时的
  NVFP4 分块一致；NVFP4 版再量化一次会叠两次误差）；lm_head 8 位每通道；激活 16 位；KV 8 位。
- **nGPT 改写**：归一化都写成 RMSNorm（权重 1/√D），slerp 的系数并进归一化权重和逐元素常量；`suv_gate`、`s_gate`、`sqk`、
  `logit_scale` 在转换时并进权重。Dory 的注意力分数到几百，掩码用 masked softmax（被掩位置取行最小值 −20），常规的加 −255
  会让被掩的旧 key 胜出。
- **循环展开**：两轮循环各用一份权重。HTP 编译器把被两个算子读的权重变成每步重算的 DDR 张量（约每参数 1 字节），循环块那一片
  因此多出 2 × 1.85 GB、超过单个上下文的上限；展开后 `.pte` 大约 1 GB（多一份循环块），每 token 读的字节数不变。
- **切 4 片**：输入段 | 第 1 轮 | 第 2 轮 | 输出段 + lm_head，各是一个 QNN 上下文。
- **KV 布局归插件管**：图把缓存当输入、只输出新算的几行 K / V，插件写回去。全局注意力的实例每个位置一个槽（槽 = 位置）；滑窗
  实例是 1088 槽的环（窗口 1024 + 64 余量，槽 = 位置 mod 1088），上下文再长也不变大。两张掩码（全局、滑窗）每步由插件按
  槽里实际是哪个位置重建（uint16，65535 = 可见）。回退只动位置；环的 64 个余量保证回退 64 个以内窗口还完整。
- **两张图共用权重**：prefill 每次 64 行；decode 每次 8 行（ExecuTorch 的 lookahead 模式，`--ngram 2 --window 8 --gcap 0`
  把 decode 图定成 8 行）。投机解码一次验证最多 8 个 token，耗时和算 1 个差不多（decode 受带宽限制）。
- **上下文档位**：静态图的 KV 长度是固定的，每个档位一个 `.pte`；全局注意力每步都按档位长度算（被掩的槽也要读），所以
  短会话用小档位更快。`npu_model` 可以列几个档位（从小到大，`…-c4096.pte,…-c32k.pte`）：**每个 prefill 块都在放得下它的
  最小档位里算**，长提示先填满小档位、再往上搬（`xtg_npu_grow`：全局缓存按位置拷前 n_past 个槽，滑窗环、位置、快照一起搬，
  不重算），对话变短时再搬回小档位；平板上搬 1056 个位置 1.5 秒，搬完继续生成的 token 和不搬时相同。同一套档位必须用同一份
  校准导出（KV 的编码要一样）。
- **注意力的两种写法**（`XTG_DORY_ATTN`）：`split` 是一个 KV 头的 4 个 q 头叠在行上、和缓存直接相乘（缓存原地读，分数
  拼接后再拆概率）；`cat` 是常见写法：缓存和新行拼起来、K / V 按 q 头复制 4 份。`cat` 在长档位每步要把整个缓存搬好几遍
  （4K 时中间结果还放得进 VTCM，长档位全走 DDR）：1 次循环 decode 每步 4K 72 ms、20K 247 ms、32K 418 ms，prefill 478 /
  73 / 50 tok/s。`split` 1 次循环 decode 每步 4K 64 ms、8K 80 ms，但 32K 680 ms（HTP 对大的分组乘法调度很差，编译估算每步读
  37 GB）；prefill 342 / 162 / 35 tok/s。所以 App 的档位是 `split` 的 4K、8K 加 `cat` 的 32K（同一份校准，KV 编码相同，
  可以互相搬）。另外插件的共享内存原来把缓存算了 3 遍（32K 时 2 GB），DSP 映射不下、QNN 每步拷贝；改成只算一遍后
  `cat` 32K 从 418 降到 280 ms。
- **HTP 跨头广播的坑**：后端的 MHA→SHA 拆头 pass 只认得 `cat` 那种图，`split` 不会被拆头。第一片之后 RoPE 的 cos / sin 是
  图的输入，HTP 会把它们跨头广播的乘法编错（编译日志 “elementwise op applied to mismatched shapes”，循环段每个注意力层
  一处），1 次循环 4K 的 NLL 1.61（`cat` 1.016）。把 `cat` 的拆头关掉（ExecuTorch `llama.py` 的 `use_mha2sha`）也一样坏，
  所以问题在广播、不在写法。`split` 时 RoPE 表先复制成 q / k 的完整形状，掩码也复制成分数的完整形状：tiny 上和 `cat` 的
  argmax 一致率从 31% 回到 96.5%。只改分组、不拆缓存的写法在设备上 prefill 直接失败（错误 1003）。

## HTP 内存

- V79（8 Elite）上单个上下文约 3.5 GiB；远映射（扩展 UDMA）要 v81，V79 没有，所以按上下文切片。
- **多上下文要用 QAIRT 2.37**（ExecuTorch 1.5.1 验证的版本）：QAIRT 2.47 下切过片的模型（包括官方 Qwen3-1.7B 切 2 片）一执行
  DSP 就崩（SSR）。
- 每片的大小可以用 `qnn-context-binary-utility` 看（`graphBlobInfo` 里的共享权重、DDR 张量、溢出缓冲、I/O）。

## 导出（服务器，x86 Linux）

```bash
git clone --branch v1.5.1 --recurse-submodules https://github.com/pytorch/executorch.git
cp -r tools/npu_et/dory executorch/examples/qualcomm/oss_scripts/dory
python tools/npu_et/apply_dory_patch.py executorch
# QNN_SDK_ROOT=<QAIRT 2.37>，ANDROID_NDK_ROOT=<NDK r27>，可编辑安装 ExecuTorch 后：
executorch/backends/qualcomm/scripts/build.sh --release        # x86 Python 绑定 + 安卓运行库、libxtg_npu.so
tools/npu_et/export_dory.sh 4096 out/c4096                       # 上下文 4096，prefill 64 行，decode 8 行
```

第一次导出时校准（6 条 agent 样本）约 12 分钟，结果存成文件（`XTG_CALIB_SAVE`）；以后各档位、各种 AR 的导出用
`XTG_CALIB_LOAD` 直接读回来（权重观测器直接看权重，不跑上下文长度的校准前向），每个档位约 40 分钟，主要是 HTP 编译。
同一份校准用两种方式导出的 `.pte` 在设备上输出逐位相同。

## 上机

模型目录放 BF16 版的 `config.json`、分词器、`chat_template.jinja`、`generation_config.json` 和 `.pte`；App 的开发版用
`-PxtgQnnLibs=<目录>` 打进 QNN 2.37 的 `libQnnHtp.so`、`libQnnSystem.so`、`libQnnHtpV79Stub.so`、`libQnnHtpV79Skel.so`，以及
`libqnn_executorch_backend.so`、`libxtg_npu.so`（QNN 的库不在仓库里，从 QAIRT 里拿）。命令行：
`xtg-server --model DIR --opt backend=npu --opt npu_model=dory-4b-npu-c4096.pte`。

```bash
# 只用插件：测速、回退一致性；批量验证和逐个算是否一致；逐位置打分（再用 compare_score.py 对参考实现）
npu_test model.pte prompt.bin 128 16
npu_test model.pte prompt.bin verify 64
npu_test model.pte prompt.bin score score.txt          # XTG_SCORE_STEP=1：逐个走 decode 图
npu_test small.pte prompt.bin grow big.pte 64          # 换档
npu_test model.pte prompt.bin kvcmp                    # prefill 和逐个算写进缓存的 KV 有多少字节不同
```

实测和精度见 [Y900 报告](reports/2026-11-08-y900-npu-dory.md)。
