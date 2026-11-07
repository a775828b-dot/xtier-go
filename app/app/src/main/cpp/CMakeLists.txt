cmake_minimum_required(VERSION 3.22)
project(xtiergo_jni CXX)

# The engine and server of this repository, built for the app's ABI (arm64-v8a), plus the JNI bridge.
set(XTG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(XTG_TESTS OFF CACHE BOOL "" FORCE)
set(XTG_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(${XTG_ROOT} xtg)

add_library(xtiergo SHARED jni.cpp)
target_link_libraries(xtiergo PRIVATE xtg_server log android)
# 16 KB pages (Android 15+ devices may use them)
target_link_options(xtiergo PRIVATE "-Wl,-z,max-page-size=16384")
