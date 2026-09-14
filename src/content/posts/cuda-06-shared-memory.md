---
title: "CUDA Programming (6): Shared Memory and Matrix Multiplication Optimization"
pubDatetime: 2026-09-02T10:00:00+08:00
description: "An in-depth guide to GPU Shared Memory, explaining its role in overcoming register limits, boosting arithmetic intensity, and accelerating GEMM through tiled matrix multiplication."
author: "SeanWang"
featured: false
draft: false
tags:
  - cuda
  - shared-memory
  - performance
  - gemm
---

> **Note**: *This article is an English translation and adaptation of the original post published on [SeanWangJS.github.io](https://seanwangjs.github.io/).*

In the previous article, we used the Roofline Model to analyze performance limits and arithmetic intensity. When computing theoretical arithmetic intensity, we made an idealized assumption: that all data required for computation could be loaded from global memory once and held in registers. 

In practice, physical register capacity per thread is strictly limited. Datasets that exceed on-chip register space must be loaded incrementally across multiple passes, often resulting in redundant memory fetches that cause the real-world arithmetic intensity to fall significantly below its theoretical ceiling.

---

## 1. The Data Movement Problem

![Figure 6.1: Data Flow Comparison: Idealized Single Load vs. Realistic Multi-Pass Fetch](/assets/posts/cuda-06-shared-memory/data_copy.png)
*Figure 6.1: Data Flow: Idealized Single Load vs. Chunked Register Loading*

Figure 6.1 illustrates how data actually moves through the processor. Because register files cannot hold an entire large tensor, the hardware must stream data in chunks into registers, compute on them, and discard them to load the next chunk. When successive phases of an algorithm reuse overlapping elements (such as sliding filters in convolutions or dot products in matrix multiplications), the same elements are re-fetched from high-latency global memory multiple times.

---

## 2. The Role of Shared Memory

To mitigate redundant high-latency DRAM roundtrips without requiring impractically vast register files, GPUs incorporate an intermediate on-chip memory tier: Shared Memory.

Shared Memory resides directly on-chip within each Streaming Multiprocessor (SM), sharing the same physical high-speed SRAM pool as the L1 data cache. However, unlike the hardware-managed L1 cache, Shared Memory is explicitly user-programmable: developers can explicitly choreograph which data to load into Shared Memory, synchronize threads, and cooperatively reuse cached data.

![Figure 6.2: Data Flow with Shared Memory as an On-Chip Buffer](/assets/posts/cuda-06-shared-memory/data_copy_with_sm.png)
*Figure 6.2: Mitigating Redundant DRAM Accesses Using On-Chip Shared Memory*

With Shared Memory acting as an on-chip software-managed cache, threads in a block cooperatively load shared data from global memory once. Even though moving data from Shared Memory into registers still occurs, the transfer is serviced across on-chip crossbars at multi-terabyte-per-second bandwidth, dramatically reducing pressure on the off-chip memory bus.

---

## 3. Optimizing Matrix Multiplication with Shared Memory

Matrix multiplication ($C = A \times B$) is the foundational linear algebra kernel powering neural networks—underpinning fully connected layers, convolutions (via `im2col` or direct GEMM), and Transformer attention projections. Optimizing GEMM provides the quintessential case study for Shared Memory tiling [[1]](#ref-1).

### 3.1. Parallel Matrix Multiplication Baseline

Consider multiplying two matrices where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $C \in \mathbb{R}^{M \times N}$. 

In CUDA, we decompose the computation into two geometric levels:
* Each thread block is responsible for calculating a $32 \times 32$ submatrix tile $C_{\text{sub}}$.
* Each thread within the block computes a single element of $C_{\text{sub}}$.

Because modern NVIDIA architectures allow up to 1024 threads per block, a $32 \times 32$ 2D block configuration fits the hardware limit ($32 \times 32 = 1024\text{ threads}$).

![Figure 6.3: Partitioning Matrix C into Blocks](/assets/posts/cuda-06-shared-memory/matmul_base.png)
*Figure 6.3: Block Partitioning for Output Matrix C*

The grid dimensions are determined by the output matrix size:
* $\text{Grid}_y = \lceil M / 32 \rceil$
* $\text{Grid}_x = \lceil N / 32 \rceil$

Within each block, a thread maps to output coordinates $(row, col)$ using the built-in variables:

```cpp
int row = blockIdx.y * 32 + threadIdx.y;
int col = blockIdx.x * 32 + threadIdx.x;
```

![Figure 6.4: Thread Mapping for Naive Matrix Multiplication](/assets/posts/cuda-06-shared-memory/matmul_thread.png)
*Figure 6.4: Thread Index Mapping for Element Computation*

The baseline naive implementation can be written as follows:

```cpp
template <typename T>
__global__ void matmul_kernel_0(const T* A, const T* B, T* C, int M, int K, int N) {
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    if (row >= M || col >= N) return;

    T sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

#### Memory Traffic Analysis of the Baseline
In this naive implementation:
* Each thread loads an entire row of $A$ ($K$ elements) and an entire column of $B$ ($K$ elements) from global memory.
* Across the entire grid, row $i$ of $A$ is reloaded $N$ times (once by every thread computing an element in row $i$ of $C$).
* Column $j$ of $B$ is reloaded $M$ times (once by every thread computing an element in column $j$ of $C$).
* Total global memory accesses scale as $2 M N K$, severely degrading effective arithmetic intensity and rendering the kernel heavily memory-bound.

---

### 3.2. Tiled Matrix Multiplication Using Shared Memory

To eliminate redundant global memory loads, we partition the inner dimension $K$ into square tiles of size $32 \times 32$. Rather than loading full rows and columns, the threads in a block cooperatively load small sub-tiles of $A$ and $B$ into Shared Memory in $\lceil K / 32 \rceil$ stages, compute partial dot products on the fast on-chip memory, and accumulate the results into registers.

![Figure 6.5: Tiled Shared Memory Matrix Multiplication](/assets/posts/cuda-06-shared-memory/matmul_thread_sm.png)
*Figure 6.5: Cooperatively Loading Tiled Submatrices into Shared Memory*

#### Memory Reduction Benefit
With $32 \times 32$ tiling:
* Every element loaded into Shared Memory is reused 32 times by other threads within the same block.
* Matrix $A$ is loaded $N / 32$ times (instead of $N$ times), and matrix $B$ is loaded $M / 32$ times (instead of $M$ times).
* Total global memory traffic is reduced by a factor of $32\times$, substantially lifting the effective arithmetic intensity towards the compute-bound roofline.

#### Tiled Kernel Implementation

```cpp
template <typename T>
__global__ void matmul_kernel_1(const T* A, const T* B, T* C, int M, int K, int N) {
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    // Allocate on-chip shared memory tiles
    __shared__ T As[32 * 32];
    __shared__ T Bs[32 * 32];

    int numTiles = (K + 32 - 1) / 32;
    T sum = 0;

    for (int i = 0; i < numTiles; ++i) {
        // Compute coordinates in global matrices
        int x_A = i * 32 + threadIdx.x;
        int y_A = blockIdx.y * 32 + threadIdx.y;
        int x_B = blockIdx.x * 32 + threadIdx.x;
        int y_B = i * 32 + threadIdx.y;

        // Cooperatively load tile into shared memory (with boundary padding)
        As[threadIdx.y * 32 + threadIdx.x] = (y_A < M && x_A < K) ? A[y_A * K + x_A] : 0;
        Bs[threadIdx.y * 32 + threadIdx.x] = (y_B < K && x_B < N) ? B[y_B * N + x_B] : 0;

        // Barrier synchronization: ensure all threads finished loading
        __syncthreads();

        // Compute partial dot product from shared memory
        for (int k = 0; k < 32; ++k) {
            sum += As[threadIdx.y * 32 + k] * Bs[k * 32 + threadIdx.x];
        }

        // Barrier synchronization: ensure calculation finishes before next tile load
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

![Figure 6.6: Coordinate Mapping for Tile Indexing](/assets/posts/cuda-06-shared-memory/A_xy_compute.png)
*Figure 6.6: Global Coordinate Calculation for Sub-Tile Extraction*

### 3.3. Key Implementation Details

1. Shared Memory Allocation (`__shared__`): The arrays `As` and `Bs` reside in on-chip Shared Memory and are shared across all 1024 threads in the block.
2. Double Barrier Synchronization (`__syncthreads`):
   * The first barrier ensures all threads have completed copying their elements from global memory into `As` and `Bs` before any thread starts computing.
   * The second barrier guarantees all threads have finished computing with the current tile before any thread overwrites `As` and `Bs` with data from the next stage.
3. Boundary Checks and Deadlock Prevention: Boundary handling must be separated from thread execution flow. In CUDA, `__syncthreads()` must be reached by all threads in the block uniformly. Exiting early with `return` when `row >= M || col >= N` causes threads to diverge before the barrier, resulting in deadlock or undefined behavior. Instead, out-of-bounds threads stay alive to participate in barriers, zero-padding shared memory during loads and omitting the final write to `C`.

### 3.4. Microarchitectural Considerations

* Bank Conflicts: Shared memory is organized into 32 memory banks. During the compute loop, all 32 threads within a warp access the same row element `As[threadIdx.y * 32 + k]`, triggering a conflict-free hardware broadcast. When accessing `Bs[k * 32 + threadIdx.x]`, thread lanes 0 through 31 access consecutive columns mapped to 32 distinct banks. Consequently, the standard $32 \times 32$ tile layout operates with zero bank conflicts.
* Arithmetic Intensity Shift: In the naive implementation, every multiply-accumulate requires reading two floats from global memory, yielding an arithmetic intensity of $2 / (2 \times 4) = 0.25\text{ FLOP/byte}$. Tiling by 32 increases arithmetic intensity by a factor of 32 to $8.0\text{ FLOP/byte}$, moving the workload significantly closer to the compute-bound roofline.
* Block Sizing and Occupancy: While a $32 \times 32 = 1024$ thread block simplifies code mapping, 1024 threads is the hardware maximum per block. In production libraries like cuBLAS or CUTLASS, kernels often use smaller thread blocks (such as $16 \times 16$ or $128$ threads) where each thread calculates a multi-element register tile (thread-level tiling), reducing shared memory pressure and increasing scheduling occupancy.

---

## References

<ul class="list-none pl-0 space-y-3">
  <li id="ref-1">
    NVIDIA Corporation, <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory" target="_blank" rel="noopener noreferrer">Shared Memory</a>, CUDA C++ Programming Guide.
  </li>
</ul>
