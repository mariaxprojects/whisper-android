cmake_minimum_required(VERSION 3.22.1)
project(bpdqwhisper C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# whisper + ggml are built static and linked into one .so
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# 1. Fetch + patch whisper.cpp (shared between ABIs and build variants)
# ---------------------------------------------------------------------------
option(BPDQ_FAST_KERNEL "Use the NEON BPDQ2 dot product instead of the patch's scalar one" ON)
set(BPDQ_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.native-deps" CACHE PATH "Where the patched whisper.cpp tree is kept")
get_filename_component(BPDQ_DEPS_DIR "${BPDQ_DEPS_DIR}" ABSOLUTE)
set(BPDQ_FAST_INC "${CMAKE_CURRENT_SOURCE_DIR}/bpdq2_fast.inc")
include("${CMAKE_CURRENT_SOURCE_DIR}/prepare_whisper.cmake")

# ---------------------------------------------------------------------------
# 2. Build whisper.cpp + ggml (CPU backend only) as static libraries
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS     OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL             OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2             OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE              OFF CACHE BOOL "" FORCE)  # cross-compiling for the phone
set(GGML_OPENMP              OFF CACHE BOOL "" FORCE)  # ggml's own thread pool; no libomp to ship
set(GGML_METAL               OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN              OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL              OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    # ARMv8.2-A with FP16 + dot product: every Cortex-A55/A75 or newer (phones from ~2018 on).
    # The app checks /proc/cpuinfo for these features before loading the library.
    set(GGML_CPU_ARM_ARCH "armv8.2-a+fp16+dotprod" CACHE STRING "" FORCE)
endif()

add_subdirectory("${WHISPER_SOURCE_DIR}" whisper-build EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# 3. JNI bridge
# ---------------------------------------------------------------------------
add_library(bpdqwhisper SHARED jni_bridge.cpp)
target_link_libraries(bpdqwhisper PRIVATE whisper ggml android log)

target_compile_options(bpdqwhisper PRIVATE -O3 -fvisibility=hidden -ffunction-sections -fdata-sections)
target_link_options(bpdqwhisper PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL
                    # 16 KB page alignment, required for apps targeting Android 15+ devices
                    -Wl,-z,max-page-size=16384)
