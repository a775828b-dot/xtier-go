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
| `packs/dory/npu.cpp` | 引擎侧：dlopen 插件、按块报进度和取消、回退、快照 |

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
- **上下文档位**：静态图的 KV 长度是固定的，每个档位一个 `.pte`，启动台按档位选；全局注意力的缓存和计算按档位长度走，
  所以短会话用小档位更快。

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
npu_test model.pte prompt.bin score score.txt
```
