# GPU 审阅快照：实现、数据与未完成项

2026-11-07，基于 `ad19fe5b414011bfe6bbb7ed8ab4e367974fe508`，按用户要求暂停优化并推送供分析。
这是开发快照，不是GPU发布版。CPU修复和此前真机验收也一并收录；未改远端main的历史。

## 结论先说清

同一Y900、325个提示token、32个生成token、8线程、2循环、last_loop、FP8 KV：

| 路线 | Prefill tok/s | Decode tok/s | 相对CPU |
|---|---:|---:|---|
| CPU NEON | 7.493 | 4.316 | 基线 |
| GPU线性层＋CPU其余算子，融合投影后 | 19.198 | 3.108 | prefill **2.56×**；decode **0.72×（慢约28%）** |

**只有prefill已获得明显加速，decode加速尚未完成。** 原方案125–250 prefill、10–13基础decode、20–35投机decode均不是本轮达到的结果。
没有把GPU设成默认，也没有在启动台开放GPU按钮。用户平板当前安装的仍是此前CPU预览版，不是这份GPU快照。

## 实现到哪一步

| 部件 | 当前状态 | 关键文件 |
|---|---|---|
| OpenCL驱动加载 | 原生arm64命令行已实际运行；失败明确报错，不静默换CPU | `engine/src/opencl.cpp` |
| NVFP4权重 | 原格式上传，E2M1＋E4M3缩放，未重新量化 | 同上 |
| GPU prefill矩阵 | 64×64输出块、K=32、每线程4×4，FP32累加 | `gemm4` |
| GPU decode矩阵 | 向量解码、子组归约（不具备时用local归约），q/k/v/g及gate/up合并调度 | `gemv4` |
| BF16输出头 | GPU GEMV/GEMM；未改FP8或NVFP4精度 | `gemv16` / `gemm16` |
| 激活/注意力/KV | **仍在CPU**，每个线性调用后读回激活 | `packs/dory/dory.cpp` |
| 性能提示 | `clSetPerfHintQCOM(HIGH)`在本机返回0；未锁定系统频率 | OpenCL初始化 |
| GPU注意力 | **只有草稿，未编译、未接线、未测数值或速度** | [`attention-draft.cl`](../experiments/attention-draft.cl) |
| 零拷贝/单份权重 | **未实现**；当前GPU约2,529,034,240字节权重副本，CPU mmap仍保留 | `CL_MEM_COPY_HOST_PTR` |
| recordable queue / 投机解码 / lm_head降精度 | **未实现** | — |
| Android App GPU入口 | Manifest已声明可选OpenCL库；**APK内GPU加载未验证，卡片仍禁用** | Manifest / card.json |

CLI选择GPU：`--opt backend=gpu`。不传该选项仍使用CPU。GPU对象归模型所有，模型卸载时释放队列、缓冲区和驱动引用。
同一组投影只上传一次输入。异步传输失败时等待队列完成再抛错，避免调用方释放仍在传输的主机内存。
当前同步CPU↔GPU边界很多，尚不是全GPU执行图。

内存估算加入GPU权重副本和临时输入/输出空间，采取保守双份计数；它不是实测峰值。不能因8GiB缓冲区实验成功就承诺长上下文可用。

## 环境与测法

- Y900 TB522FU，SM8750P / Adreno 830，Android 16，系统可管理约15GiB。连接电源；未清空系统服务、未锁频、没有持续频率采样。
- OpenCL 3.0，QUALCOMM 0800.72。GPU测试在 `adb shell` 原生程序里运行，不能替代APK命名空间/权限验证。
- NDK 28.2.13676358，arm64-v8a，android-29，Release，c++_static。
- 模型：卡片固定的Dory NVFP4 revision `64b4039a3468bef85c6744c1e24d5a80801fe965`，SHA256 `ca08dec5d70b653c03e57903dd33725ec8cc80133c53a1bd9c83f99dd03f3e46`。
- 激活和累加为FP32；使用OpenCL `mad`，不能宣称与CPU严格FMA逐位相同。
- 整模型速度每配置仅一次，不是交错多轮统计。短时温度和调度会影响结果，不据此承诺持续性能。

## 1. GPU内核数值与测速

`opencl_test`覆盖NVFP4的M=1/2/7/65/512，N×K=70×256、7168×2560、2560×7168；BF16的M=1/2/16，N×K=131072×2560。
还检查4投影融合与单投影一致、非整块边界。每个计时组先预热一次、再测3次取中位数。
精度对CPU double点积抽样（每矩阵首/末两行各32列），最大绝对误差除以参考RMS，阈值3e-5；**不是整矩阵逐元素检查**。

以下取最终融合版本的记录：[kernels-fused.txt](data/y900-20261107/kernels-fused.txt)。

| 权重/用途 | M×N×K | 毫秒 | GFLOP/s | RMS归一化最大抽样误差 |
|---|---|---:|---:|---:|
| NVFP4 decode | 1×7168×2560 | 0.980 | 37.44 | 2.24e-7 |
| NVFP4 prefill | 512×7168×2560 | 53.375 | 352.05 | 2.83e-6 |
| NVFP4 decode | 1×2560×7168 | 0.670 | 54.78 | 2.81e-7 |
| NVFP4 prefill | 512×2560×7168 | 54.856 | 342.54 | 7.07e-6 |
| BF16输出头decode | 1×131072×2560 | 11.431 | 58.71 | 3.06e-7 |
| BF16输出头多行 | 16×131072×2560 | 126.974 | 84.56 | 1.82e-6 |

耗时包含本次调用的激活上传、GPU运算、结果读回和同步，权重准备在计时前。GPU只测小矩阵快，不代表整模型一定快。

## 2. 整模型数值对照

第一次线性后端的96token前向，对照CPU与GPU：2循环、last_loop、FP8 KV、context=4096、prefill_chunk=32，完整词表logits输出。
输入混合英文与中文，具体token见 [ids.json](data/y900-20261107/ids.json)。

| 指标 | 实测 |
|---|---:|
| 96位置argmax一致率 | 100% |
| 下一token logprob平均绝对差 | 0.00003327 |
| 全部logits平均绝对差 | 0.002745 |
| 全部logits最大绝对差 | **0.498021** |
| CPU前向耗时 | 13.777秒 |
| GPU前向耗时 | 10.920秒 |

原始汇总：[comparison.json](data/y900-20261107/comparison.json)；逐位置结果：[cpu.json](data/y900-20261107/cpu.json)、[gpu.json](data/y900-20261107/gpu.json)。
两份约50MiB的logits二进制留在开发工作区，没有提交到仓库。

**最大logit差不满足原FP32参考的1e-3门槛。** argmax与目标token logprob接近，不能证明所有输出、长上下文、工具任务都等价。
这是第一版前向测试；后续decode融合/向量化通过内核测试，但尚未重新做完整前向、PPL、逐层及全部循环模式对照。
因Dory注意力敏感，仍须定位最大差异与质量影响，不能将这个快照描述为精度验收完成。

## 3. 整模型速度与每轮改动

输入token固定见 [generate-ids.json](data/y900-20261107/generate-ids.json)：325token；原始文本为重复的英中混合说明，非chat模板、非DSH真实agent请求。
`xtg-cli generate`，greedy、2循环、last_loop、FP8、context4096、prefill_chunk512。

| 轮次 | CPU线程 | 输出token | Prefill | Decode | 说明 |
|---|---:|---:|---:|---:|---|
| CPU基线 | 8 | 32 | 7.493 | 4.316 | NEON |
| GPU向量decode、未融合 | 8 | 32 | 19.436 | 3.098 | prefill快，decode慢 |
| GPU融合q/k/v/g及gate/up | 8 | 32 | 19.198 | 3.108 | 融合未带来明显整模型改善 |
| 加HIGH提示与profile | 8 | 16 | 19.391 | 2.987 | 输出长度变了，不直接与32token比高低 |
| 同路线减少CPU线程 | 4 | 16 | 18.385 | 3.659 | 仍未达到CPU基线 |
| 同路线单CPU线程 | 1 | 16 | 14.012 | 1.655 | 明显退化，不采用 |

单位都是token/s。32tokenCPU/GPU输出与速度见 [cpu-generate.json](data/y900-20261107/cpu-generate.json)、[gpu-generate.json](data/y900-20261107/gpu-generate.json)。未融合轮没有另留JSON副本，表中值来自当次终端输出；融合后的JSON是保留文件的版本。
这组CPU/GPU的32个输出token ID完全相同，仅说明该样本一致。
CLI没有服务级预热：GPU权重上传发生在prefill计时之前，CPU的mmap路径可能在首次计算时缺页；未做交错、多轮、相同热态的性能验收。2.56倍是本次观察值，不是已排除预热偏差后的稳定加速比。
三个profile原始输出：[8线程](data/y900-20261107/profile.txt)、[4线程](data/y900-20261107/profile4.txt)、[1线程](data/y900-20261107/profile1.txt)。

**CLI计数口径的限制：** 当前generate循环还计算了最后一个输出之后不需要的一步，但速度分子用`输出数−1`。因此其绝对decode值略偏低；同一脚本的CPU/GPU比较口径一致。本次没修改这个原有计数逻辑，避免混淆前后比较。不要直接对比HTTP服务或外部引擎的tok/s。

此前CPU App HTTP实测是另外两组133/325提示、各64输出、关闭思考的chat请求：6.66–7.19 prefill、3.68–4.38 decode。见 [CPU基线JSON](data/y900-20261107/cpu-baseline.json)；**不是本表的同一负载**。

## 4. 耗时拆分：目前知道什么

16输出的profiling：

| CPU线程 | NVFP4 decode总耗时/调用 | BF16头总耗时/调用 | CPU注意力＋KV store |
|---:|---|---|---|
| 8 | 2.682秒 / 3584 | 0.209秒 / 17 | 未单独记录 |
| 4 | 2.694秒 / 3584 | 0.214秒 / 17 | 1.150秒 |
| 1 | 4.253秒 / 3584 | 0.221秒 / 17 | 4.286秒 |

NVFP4每步224次线性调用；融合减少的是一个调用内的投影kernel数量，**没有消除这些CPU/GPU同步边界**。
BF16统计含prefill取最后位置logits的一次调用，不能把整列都当纯decode。上述时间包括各方法主机侧上传、等待、读回，尚未用GPU事件逐项拆开。
单线程同时拖慢CPU注意力与主机调度，不能仅依据CPU线程数或带宽给出速度承诺。

## 5. 已撤回与隔离的实验

- 将prefill加载改为连续K方向并给local tile加padding，实测从约342–352跌到约200GFLOP/s；已撤回。数据：[kernels-coalesced.txt](data/y900-20261107/kernels-coalesced.txt)。
- 标量NVFP4 decode向量化后，真实矩阵单次从约1.9–2.0ms降到约0.65–0.72ms；这是内核改善，不等于整模型decode改善。数据：[原版](data/y900-20261107/kernels-warm.txt)、[向量版](data/y900-20261107/kernels-vector2.txt)。
- 注意力草稿单独放在`docs/experiments/attention-draft.cl`。它还依赖未实现的GPU KV镜像、增量上传、回退/快照失效逻辑；未纳入已测试程序，勿直接启用。

## 6. 可复现命令

```bash
cmake -S . -B build-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static
cmake --build build-android -j4
adb push build-android/opencl_test build-android/xtg-cli /data/local/tmp/xtg-gpu/
adb shell chmod 700 /data/local/tmp/xtg-gpu/opencl_test /data/local/tmp/xtg-gpu/xtg-cli
adb shell /data/local/tmp/xtg-gpu/opencl_test

# 把下面模型目录换成设备上实际路径，输入是本报告附带的generate-ids.json。
adb push docs/reports/data/y900-20261107/generate-ids.json /data/local/tmp/xtg-gpu/
adb shell /data/local/tmp/xtg-gpu/xtg-cli generate --model MODEL_DIR \
  --ids /data/local/tmp/xtg-gpu/generate-ids.json --threads 8 --max 32 --temp 0 \
  --opt loops=2 --opt kv_mode=last_loop --opt kv_dtype=fp8 --opt context=4096 \
  --opt prefill_chunk=512 --opt backend=gpu
# CPU对照只将backend=gpu换为backend=cpu；profile可在设备命令前设置XTG_GPU_PROFILE=1。
```

当前模型卡的backend仍为launch字段且GPU被禁用，App不会把它传成模型option；这一步有意留到GPU验收后再开放。CLI可以显式选择GPU。

## 7. 已通过和未通过

- CPU初版修复：真机5项文件/启停测试通过，真实模型本地导入校验通过，现有API检查通过，原生内核/解析器通过。对应日志放在本报告data目录；独立外网镜像下载超时，没有宣称通过。
- GPU：原生命令行编译、18组矩阵形状和融合边界测试通过；短提示前向argmax/logprob对照如上。
- 推送快照前：隔离注意力草稿后重新编译原生工具和debug APK成功，真机GPU矩阵、NEON与解析器再次通过，日志见 [snapshot-smoke.txt](data/y900-20261107/snapshot-smoke.txt)。该次M=512的NVFP4约314–332GFLOP/s，显示不同轮次存在波动；没有替换平板已安装的CPU APK。
- **未完成：** decode收益、最大logit差解释/完整质量验收、GPU注意力、APK内GPU执行、DSH实际任务、长上下文、持续热稳定性、单份权重、recordable queue与投机解码。

用户希望先审阅当前实现，因此本次暂停在这里。建议先分析同步边界和CPU注意力耗时，以及是否允许一个“GPU prefill＋CPU decode”的过渡模式；尚未将该过渡模式当成完成的GPU decode加速。
