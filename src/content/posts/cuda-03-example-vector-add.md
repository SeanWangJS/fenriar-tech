---
title: "CUDA Programming (3): A Vector Addition Example"
pubDatetime: 2026-08-30T10:00:00+08:00
description: "A practical hands-on guide to implementing vector addition in CUDA C++, covering memory allocation, kernel invocation syntax, and compilation with nvcc."
author: "SeanWang"
featured: false
draft: false
tags:
  - cuda
  - gpu
  - parallel-computing
---

> **Note**: *This article is an English translation and adaptation of the original post published on [SeanWangJS.github.io](https://seanwangjs.github.io/).*

In the preceding articles, we explored the hardware architecture of NVIDIA GPUs and the hierarchical CUDA programming model built on top of it. Now, let us walk through a practical vector addition example to demonstrate how to write a parallel computing program using CUDA C++ extensions.

---

## 1. Typical CUDA Workflow

From a system architecture perspective, a CUDA application involves execution across both the CPU and GPU. In CUDA terminology, the CPU and its system memory are referred to as the **Host**, while the GPU and its onboard memory are referred to as the **Device**. A canonical CUDA workflow proceeds through the following steps:

1. **Allocate memory on the Host**, populate input data, allocate corresponding memory on the Device, and transfer the input data from Host to Device.
2. **Define the parallel algorithm on the Device** in the form of a CUDA kernel function.
3. **Launch the kernel from the Host**, configuring the execution grid and block dimensions to execute the computation across thousands of threads.
4. **Wait for kernel execution to complete**, and transfer the computed results back from Device memory to Host memory.

---

## 2. Complete Vector Addition Implementation

The following code illustrates a complete, self-contained vector addition program:

```cpp
// main.cu

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

// Number of threads per block
#define BLOCK_SIZE 256

// CUDA kernel executing on the Device
__global__ void vectorAdd(const float* a, 
                          const float* b,
                          const int n,
                          float* c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    // Array length
    int n = 1 << 20;
    // Calculate the number of blocks needed
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate host memory
    size_t size = n * sizeof(float);
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);
    float* c_ref = (float*)malloc(size); // Reference output on Host for verification

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host arrays to device memory
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, n, d_c);

    // Copy result from device back to host (cudaMemcpy blocks the Host until the preceding kernel finishes)
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results against CPU reference
    for (int i = 0; i < n; i++) {
        c_ref[i] = a[i] + b[i];
        if (c[i] != c_ref[i]) {
            printf("Error: c[%d] = %f, c_ref[%d] = %f\n", i, c[i], i, c_ref[i]);
            break;
        }
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    free(c_ref);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

---

## 3. Key Details Explained

Several aspects of this program warrant closer examination:

1. **The `__global__` Execution Space Specifier**: Functions that execute on the Device and are callable from the Host must be declared with the `__global__` specifier and must return `void`, signaling to the compiler that this function is a CUDA kernel entry point.
2. **Device Memory Management & Stream Synchronization**: Allocating and freeing memory on the Device is handled through dedicated CUDA runtime APIs (`cudaMalloc` and `cudaFree`), while data transfers between Host and Device are coordinated using `cudaMemcpy` with explicit direction flags (`cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`). Because standard `cudaMemcpy` is synchronous with respect to the Host and stream-ordered after the kernel in the default stream, the Host automatically blocks until data transfer completes, making an explicit `cudaDeviceSynchronize()` redundant prior to reading the results.
3. **Kernel Launch Syntax**: The standard syntax for invoking a kernel is `func<<<dimGrid, dimBlock>>>(args...)`. Unlike regular C++ function calls, the execution configuration is passed between triple angle brackets (`<<<...>>>`). These parameters configure the execution geometry: `dimGrid` defines the dimensions and number of blocks in the grid, while `dimBlock` defines the dimensions and number of threads within each block. Both can be specified as 3D structures (`dim3`), spanning the x, y, and z axes. If fewer dimensions are provided (such as an integer scalar), the unspecified dimensions implicitly default to 1; for instance, `dimGrid = 10` is equivalent to `dimGrid = (10, 1, 1)`.

---

## 4. Compilation with nvcc

To compile CUDA programs, you use NVIDIA's `nvcc` compiler driver, which is bundled directly with the CUDA Toolkit. On Linux systems, the default installation path is `/usr/local/cuda/bin`, while on Windows, it typically resides in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\bin`.

The compilation command is straightforward:

```bash
nvcc main.cu -o main
```

Because `nvcc` automatically links against essential CUDA runtime libraries and includes standard header search paths, you do not need to manually specify CUDA header directories or runtime library flags for basic programs.
