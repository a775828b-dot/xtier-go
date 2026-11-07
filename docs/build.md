# 构建

[回到文档目录](README.md)

依赖都在仓库里（`third_party/`：nlohmann/json 3.12.0、cpp-httplib 0.65.0、minja 021c229，许可见 [NOTICE](../NOTICE)），
不需要联网。

## Linux（开发、对照）

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build                      # xtg-server、xtg-cli、kernels_test、parsers_test
build/kernels_test && build/parsers_test
build/xtg-server --model /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --threads 32
```

x86-64 默认 `-march=x86-64-v3`（AVX2 / FMA，`-DXTG_HOST_ARCH=` 关掉）。x86 上只有通用内核，用来对照正确性，不追求速度
（服务器 64 线程：prefill 约 50 tok/s，decode 约 12–15 tok/s）。

## aarch64 Linux + QEMU（不用手机测 NEON 内核）

```bash
sudo apt-get install g++-aarch64-linux-gnu qemu-user
cmake -S . -B build-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
ninja -C build-arm64 kernels_test parsers_test
qemu-aarch64 -L /usr/aarch64-linux-gnu build-arm64/kernels_test   # 会比较 NEON 和通用内核
```

## Android 命令行工具（adb shell 里跑，不用装 App）

```bash
cmake -S . -B build-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static
ninja -C build-android
adb push build-android/xtg-server build-android/xtg-cli build-android/kernels_test /data/local/tmp/
adb push <模型目录> /data/local/tmp/dory/
adb shell /data/local/tmp/kernels_test
adb shell /data/local/tmp/xtg-server --model /data/local/tmp/dory --threads 8
adb forward tcp:18400 tcp:18400      # 电脑上的脚本就能连 127.0.0.1:18400
```

CI（`.github/workflows/check.yml` 的 `android-tools`）每次推送都编一份，Actions 页面的 artifact `xtg-android-arm64-tools`
可以直接下载。

## App

```bash
cd app && gradle assembleDebug      # 或用 Android Studio 打开 app/
adb install app/build/outputs/apk/debug/app-debug.apk
```

要 JDK 17、Android SDK（compileSdk 35）、NDK 27.2.12479018、SDK 里的 CMake 3.22.1。App 的 native 部分
（`app/app/src/main/cpp/CMakeLists.txt`）用 `add_subdirectory` 编整个仓库，只编 arm64-v8a。仓库里没有 Gradle wrapper，
CI 用 `gradle/actions/setup-gradle` 装 Gradle 8.11.1；本地第一次可以 `gradle wrapper --gradle-version 8.11.1` 生成。
CI 的 `android-app` 任务每次推送都编一份 debug APK（artifact `xtier-go-debug-apk`）。

模型文件放 `Android/data/io.github.a775828b.xtiergo/files/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/`：启动台里可以从
hf-mirror / huggingface 直连下载（域名经 223.5.5.5 解析、不走代理，8 连接分段、断点续传、下完校验 sha256），也可以 `adb push` 进去。

### 带 QNN 的开发版（NPU 自检）

`-PxtgQnnLibs=DIR` 把 `DIR/arm64-v8a/` 一起打进 APK，并在安装时解出原生库，这样 App 才能执行其中两个程序。目录里放 QNN SDK
（QAIRT 2.47）的这些文件：

- `lib/aarch64-android/` 下的 `libQnnHtp.so`、`libQnnHtpV79Stub.so`、`libQnnSystem.so`、`libQnnHtpProfilingReader.so`；
- `lib/hexagon-v79/unsigned/` 下的 `libQnnHtpV79Skel.so`、`libQnnHtpV79.so`；
- `bin/aarch64-android/` 下的 `qnn-net-run`、`qnn-profile-viewer`，分别改名成 `libqnn_net_run.so`、`libqnn_profile_viewer.so`。

SDK 不在仓库里，这个选项只用于本地构建。

```bash
gradle assembleDebug -PxtgAppIdSuffix=.dev -PxtgQnnLibs=/path/to/qnn-jnilibs
# 上下文二进制（*_m<M>.bin）、list_m<M>.txt 和 inputs/ 放进 App 的 files/npu/（tools/npu_probe 的 kit 里有）
adb shell am start -n io.github.a775828b.xtiergo.dev/io.github.a775828b.xtiergo.MainActivity --ez npu_selftest true
adb shell cat /sdcard/Android/data/io.github.a775828b.xtiergo.dev/files/npu/selftest.txt
```

DEV 面板的「NPU 自检」按钮做的是同一件事。App 自己打不开 /dev/fastrpc-cdsp，这是正常的：libcdsprpc.so 会经 vendor 的
dspservice 拿设备。见[摸底报告](reports/2026-11-07-y900-npu-gpu-probe.md#n0app-进程里的-npu11-07-夜)。

### 本地构建与真机回归

已有其他 NDK 时可用 `-PxtgNdkVersion=28.2.13676358` 覆盖；未指定仍使用原来的 27.2.12479018。
Y900 本轮使用 Gradle 8.10.2、NDK 28.2、JDK 17，通过 `assembleDebug assembleDebugAndroidTest lintDebug`。

安装主 APK 和 androidTest APK 后：

```bash
adb shell am instrument -w -r -e class io.github.a775828b.xtiergo.ModelStoreTest,io.github.a775828b.xtiergo.EngineServiceTest \
  io.github.a775828b.xtiergo.test/androidx.test.runner.AndroidJUnitRunner
```

文件测试使用独立临时目录；服务测试需要卡片中的真实模型，否则跳过。服务测试会启动、停止当前本机服务，验证长 prefill 与排队请求退出、主线程响应以及重启。
`ModelDownloadTest` 仅显式传入 `-e modelMirror <URL>` 时下载真实模型；本轮 USB 本地服务只验证传输和哈希，不代表独立联网下载通过。

Y900 上用 ADB 提前创建 external-files 下的模型目录会得到 shell 所有者，App 可能无法写入。让 App 创建目录并通过自己的下载器写入；
不要以放宽整个目录权限作为修复。导入已有文件时，带哈希的文件必须校验后才能启动。
