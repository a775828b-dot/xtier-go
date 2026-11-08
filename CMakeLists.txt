cmake_minimum_required(VERSION 3.22)
project(xtier_go CXX)

# The engine, the model packs and the server as libraries; xtg-server / xtg-cli for Linux (and adb shell), the
# Android app builds the same sources into libxtiergo.so (app/app/src/main/cpp).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(XTG_TESTS "build the unit tests" ON)
option(XTG_TOOLS "build xtg-server and xtg-cli" ON)
# x86-64 hosts are development machines: AVX2 / FMA make the portable kernels usable for checks
set(XTG_HOST_ARCH "x86-64-v3" CACHE STRING "-march for x86-64 builds (empty: compiler default)")

add_compile_options(-Wall -Wextra -Wno-unused-parameter -fopenmp-simd)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND XTG_HOST_ARCH)
  add_compile_options(-march=${XTG_HOST_ARCH})
endif()

find_package(Threads REQUIRED)

add_library(xtg_engine STATIC
  engine/src/common.cpp
  engine/src/fp.cpp
  engine/src/threadpool.cpp
  engine/src/safetensors.cpp
  engine/src/kernels.cpp
  engine/src/kernels_neon.cpp
  engine/src/opencl.cpp
  engine/src/kv.cpp
  engine/src/sampler.cpp
  engine/src/model.cpp
  packs/packs.cpp
  packs/dory/dory.cpp
  packs/dory/npu.cpp)
target_include_directories(xtg_engine PUBLIC engine/include third_party PRIVATE packs npu third_party/OpenCL)
target_link_libraries(xtg_engine PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(ANDROID)
  target_link_libraries(xtg_engine PUBLIC log)
endif()

add_library(xtg_server STATIC
  server/tokenizer.cpp
  server/chat.cpp
  server/parsers.cpp
  server/runtime.cpp
  server/http.cpp
  server/capi.cpp)
target_include_directories(xtg_server PUBLIC server)
target_link_libraries(xtg_server PUBLIC xtg_engine)

if(XTG_TOOLS)
  add_executable(xtg-cli tools/xtg_cli.cpp)
  target_link_libraries(xtg-cli PRIVATE xtg_server)
  add_executable(xtg-server tools/xtg_server.cpp)
  target_link_libraries(xtg-server PRIVATE xtg_server)
endif()

if(XTG_TESTS)
  enable_testing()
  add_executable(kernels_test tests/kernels_test.cpp)
  target_link_libraries(kernels_test PRIVATE xtg_engine)
  add_test(NAME kernels COMMAND kernels_test)
  add_executable(parsers_test tests/parsers_test.cpp)
  target_link_libraries(parsers_test PRIVATE xtg_server)
  add_test(NAME parsers COMMAND parsers_test)
  # Explicit device test: normal CPU CI does not require an OpenCL GPU.
  add_executable(opencl_test tests/opencl_test.cpp)
  target_link_libraries(opencl_test PRIVATE xtg_engine)
  # GPU building blocks on the device (launch overhead, bandwidth, multi-row NVFP4 GEMV, peaks); not run by CI
  add_executable(gpu_probe tests/gpu_probe.cpp)
  target_include_directories(gpu_probe PRIVATE third_party/OpenCL)
  target_link_libraries(gpu_probe PRIVATE xtg_engine)
endif()
