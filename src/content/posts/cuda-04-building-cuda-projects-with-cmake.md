---
title: "CUDA Programming (4): Building CUDA Projects with CMake and GoogleTest"
pubDatetime: 2026-08-31T10:00:00+08:00
description: "A step-by-step guide to structuring a professional CUDA C++ project using modern CMake, configuring VS Code IntelliSense, and integrating GoogleTest for automated testing."
author: "SeanWang"
featured: false
draft: false
tags:
  - cuda
  - cmake
  - cplusplus
  - tooling
---

> **Note**: *This article is an English translation and adaptation of the original post published on [SeanWangJS.github.io](https://seanwangjs.github.io/).*

In the previous article, we wrote a standalone vector addition program compiled with a single command. However, production-grade applications require a modular project layout, automated build systems, and unit testing suites. In this article, we demonstrate how to configure a multi-file CUDA C++ project using modern CMake, set up VS Code for IntelliSense, and integrate GoogleTest for automated testing.

---

## 1. Project Directory Structure

A clean, modular layout separates public headers, kernel implementations, build definitions, and test suites:

```text
├── .vscode/
│   └── c_cpp_properties.json
├── include/
│   └── vector_add.h
├── src/
│   └── vector_add_kernel.cu
├── test/
│   ├── CMakeLists.txt
│   └── test.cpp
└── CMakeLists.txt
```

The `.vscode/c_cpp_properties.json` file configures IntelliSense for the VS Code C/C++ extension, enabling the editor to resolve CUDA and project headers specified in `includePath`:

```json
{
  "configurations": [
    {
      "name": "Win32",
      "intelliSenseMode": "windows-msvc-x64",
      "cStandard": "c17",
      "cppStandard": "c++17",
      "includePath": [
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/include",
        "${workspaceFolder}/include"
      ]
    }
  ],
  "version": 4
}
```

> **Tip**: If you use the VS Code CMake Tools extension, you can set `"configurationProvider": "ms-vscode.cmake-tools"` inside `c_cpp_properties.json`. This allows VS Code to automatically synchronize include paths and compile definitions directly from your CMake targets without hardcoding local file paths.

---

## 2. Root CMakeLists.txt

Modern CMake (version 3.8 and above) treats CUDA as a first-class language alongside C and C++. Instead of relying on the legacy, deprecated `find_package(CUDA)` module, we enable CUDA directly through the `project` declaration:

```cmake
cmake_minimum_required(VERSION 3.18)

project(vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile CUDA sources into a static library
add_library(vector_add STATIC
    src/vector_add_kernel.cu
)

# Target include directories automatically propagate to downstream consumers
target_include_directories(vector_add PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable CTest and integrate GoogleTest
enable_testing()

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # Align GoogleTest runtime linkage with MSVC defaults on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(test)
```

Key aspects of this configuration:
* `LANGUAGES CXX CUDA`: Instructs CMake to compile `.cu` files using `nvcc` and `.cpp` files with the host C++ compiler natively, without legacy wrapper macros.
* Target Usage Requirements (`PUBLIC`): By specifying `target_include_directories(vector_add PUBLIC ...)`, any downstream target that links against `vector_add` automatically inherits the `include/` path.
* Robust GoogleTest Discovery: It checks for a system-installed GoogleTest first; if absent, it leverages CMake's `FetchContent` to download and compile GoogleTest automatically.

---

## 3. Test Suite CMakeLists.txt

Inside `test/CMakeLists.txt`, we configure the test runner executable and register it with CTest:

```cmake
cmake_minimum_required(VERSION 3.18)

add_executable(test_vector_add test.cpp)

# Target linking automatically pulls in include paths from vector_add and GoogleTest
target_link_libraries(test_vector_add PRIVATE
    vector_add
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_vector_add)
```

Notice how clean modern target-based CMake is:
* Linking `vector_add` automatically propagates the project's header include directory without manual path definitions.
* Linking `GTest::gtest_main` automatically provides the default test entry point (`main()`) and requisite compiler flags.
* `gtest_discover_tests` queries the compiled binary and registers individual test cases directly with CTest.

---

## 4. Build and Execution

With CMake installed, we build and run the project using standard out-of-source build commands:

```bash
# 1. Configure the build directory
cmake -B build -S .

# 2. Compile all targets
cmake --build build --config Release

# 3. Run unit tests via CTest
ctest --test-dir build -C Release --output-on-failure
```

If you prefer to manually supply precompiled GoogleTest paths rather than using automated fetching, you can pass them via `-DGTEST_ROOT` or standard package path hints during the configuration step:

```bash
cmake -B build -S . -DGTEST_ROOT=/path/to/googletest
```

---

## 5. Resolving MSVC Runtime Library Mismatches on Windows

On Windows, developers compiling against GoogleTest frequently encounter linker errors regarding runtime library mismatches:

```text
error LNK2038: mismatch detected for 'RuntimeLibrary': value 'MTd_StaticDebug' doesn't match value 'MDd_DynamicDebug' in ...
```

This occurs because Visual Studio builds default to dynamic C runtime linking (`/MD` or `/MDd`), whereas precompiled GoogleTest static libraries may have been built against the static C runtime (`/MT` or `/MTd`).

When compiling GoogleTest from source (or via `FetchContent`), passing `-Dgtest_force_shared_crt=ON` [[1]](#ref-1) ensures that GoogleTest matches the project's dynamic MSVC runtime linkage.

---

## References

<ul class="list-none pl-0 space-y-3">
  <li id="ref-1">
    Google, <a href="https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes" target="_blank" rel="noopener noreferrer">Visual Studio Dynamic vs. Static Runtimes</a>, GoogleTest Documentation.
  </li>
</ul>
