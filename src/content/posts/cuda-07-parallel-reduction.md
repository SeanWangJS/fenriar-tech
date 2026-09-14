---
title: "CUDA Programming (7): Parallel Reduction and Kernel Optimization Techniques"
pubDatetime: 2026-09-03T10:00:00+08:00
description: "A comprehensive case study on optimizing parallel reduction in CUDA, covering Roofline modeling, control divergence, shared memory bank conflicts, loop unrolling, warp shuffle instructions, and thread coarsening."
author: "SeanWang"
featured: false
draft: false
tags:
  - cuda
  - parallel-computing
  - optimization
  - reduction
  - roofline
---

> **Note**: *This article is an English translation and adaptation of the original post published on [SeanWangJS.github.io](https://seanwangjs.github.io/), inspired by Mark Harris' classic presentation on parallel reduction [0].*

CUDA programming encompasses multiple conceptual layers: GPU microarchitecture, the CUDA programming and execution model, and parallel algorithm design. Developing a thorough understanding of these concepts enables writing high-performance CUDA programs. Taking the canonical parallel reduction operator—array summation—as a case study, this article walks through common kernel optimization techniques and demonstrates how each contributes to performance gains.

---

## 1. Roofline Model Analysis

Before implementing an algorithm, analyzing its theoretical upper bound on the target hardware using the Roofline model provides a realistic performance expectation. For a single-precision floating-point (FP32) array summation of size $n$:
* Total computation: $n - 1$ additions ($\approx n\text{ FLOPs}$).
* Total memory traffic: $4n$ bytes read.
* Theoretical arithmetic intensity: approximately $0.25\text{ FLOP/byte}$.

Taking an NVIDIA GeForce RTX 4050 Laptop GPU as the testing baseline:
* Peak FP32 throughput: 8.986 TFLOPS [1].
* Peak memory bandwidth: 192 GB/s.

Plotting its Roofline model:

![Figure 7.0: Roofline Model for Reduction Operator on RTX 4050](/assets/posts/cuda-07-parallel-reduction/rtx4050_roofline.png)
*Figure 7.0: Roofline model for reduction on NVIDIA RTX 4050 Laptop GPU*

The knee point coordinate of 46.80 FLOP/byte is obtained from $8986 / 192$. The black dot marks the theoretical performance ceiling for reduction on this GPU: when saturating 100% of memory bandwidth, the achievable compute throughput is approximately 48 GFLOPS ($192\text{ GB/s} \times 0.25\text{ FLOP/byte}$).

---

## 2. Naive Reduction via Atomic Operations

To sum $N$ elements, the most straightforward approach is to declare a global variable and have each thread load one element from the array and add it to this accumulator. To prevent data races across threads, we use CUDA's built-in `atomicAdd`:

```cpp
template <typename T>
__global__ void reduce_kernel_0(const T* data, const size_t n, T* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    atomicAdd(&result[0], data[tid]);
}
```

While `atomicAdd` ensures correct synchronization without data races, serialized access to a single global memory address incurs a time complexity equivalent to sequential execution, resulting in poor throughput [2].

---

## 3. Tree-Based Reduction Pattern

To leverage the massive parallelism of GPU cores, a divide-and-conquer strategy is preferred: partition the array into blocks, compute the sum of each block, and combine the results. Within each block, recursive partitioning produces a reduction tree [3]:

![Figure 7.1: Tree-Based Reduction Pattern](/assets/posts/cuda-07-parallel-reduction/tree_reduce.png)
*Figure 7.1: Tree-based reduction pattern*

In Figure 7.1, green boxes represent elements participating in computation at each stage. In each stage, every active core adds two adjacent elements without requiring global locks. Synchronization is required between stages to ensure all cores have completed their additions before proceeding to the next stage. Eventually, all values within the block are accumulated into the first element.

In CUDA, each thread acts as an execution core. Since the number of threads per block is bounded (up to 1024), larger arrays require launching multiple blocks. The overall parallel reduction workflow comprises two steps:
1. Perform tree reduction locally within each block.
2. Accumulate the per-block results into the final total sum.

The baseline tree reduction kernel can be written as follows:

```cpp
template <typename T, int blockSize>
__global__ void reduce_kernel_1(const T* data, const size_t n, T* result) {
    __shared__ T sdata[blockSize];

    int tid = threadIdx.x;
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    sdata[tid] = (globalId >= n) ? 0 : data[globalId];

    __syncthreads();

    for (int s = 1; s < blockSize; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&result[0], sdata[0]);
    }
}
```

### Walkthrough of Baseline Kernel
1. `data` is the input array, `n` is element count, and `result` stores the accumulated sum.
2. A shared memory buffer `sdata` of size `blockSize` is allocated per block.
3. Each thread reads its element from global memory into `sdata`. If `globalId` exceeds `n`, the element is padded with zero.
4. `__syncthreads()` acts as a block-wide barrier until all threads finish loading data into shared memory.
5. In the stage loop with step variable `s`, only threads satisfying `tid % (2 * s) == 0` participate:
   * When `s = 1`, threads `0, 2, 4, 6, ...` are active.
   * When `s = 2`, threads `0, 4, 8, 12, ...` are active.
   * Active thread `tid` adds `sdata[tid + s]` into `sdata[tid]`. A barrier `__syncthreads()` synchronizes threads between stages. The total number of stages is $\log_2(\text{blockSize})$.
6. After all stages complete, `sdata[0]` holds the block sum. Thread 0 adds `sdata[0]` to `result[0]` using `atomicAdd`.

The corresponding host harness code:

```cpp
#define BLOCK_SIZE 256

template <typename T>
T reduce(T* data, 
         size_t n, 
         int type, 
         int numBlocks) {
    T* d_data;
    T* d_result;
    T result[1] = {0};

    cudaMalloc((void**)&d_data, sizeof(T) * n);
    cudaMalloc((void**)&d_result, sizeof(T) * numBlocks);

    cudaMemcpy(d_data, data, sizeof(T) * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int repeat = 100;
    for (int i = 0; i < repeat; i++) {
        cudaMemcpy(d_result, result, sizeof(T) * 1, cudaMemcpyHostToDevice);
        switch(type) {
            case 1:
                reduce_kernel_1<T, BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(d_data, n, d_result);
                break;
            default:
                std::cout << "Invalid type: " << type << std::endl;
                exit(1);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total time: " << milliseconds << "ms" << std::endl;

    size_t bytes = sizeof(T) * n * repeat;
    float seconds = milliseconds / 1000;
    float throughput = bytes / seconds / 1000 / 1000 / 1000;
    std::cout << "Throughput: " << throughput << "GB/s" << std::endl;

    size_t flops = n * repeat;
    float gflops = flops / seconds / 1000 / 1000 / 1000;
    std::cout << "Performance: " << gflops << "GFLOPS" << std::endl;

    cudaMemcpy(result, d_result, sizeof(T) * 1, cudaMemcpyDeviceToHost);
    return result[0];
}
```

Baseline performance on $N = 2^{24}$ elements (16M floats):

| Version | Total Time (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 124.30 | 53.99 | 13.50 | 1.00 |

---

## 4. Profiling Setup with Nsight Compute [4]

NVIDIA Nsight Compute (NC) is the dedicated kernel profiling tool for CUDA programs. Launch Nsight Compute with administrator privileges and select **Connection -> Connect**:

![Figure 7.2: Nsight Compute Connection Window](/assets/posts/cuda-07-parallel-reduction/nsight_compute.png)
*Figure 7.2: Nsight Compute connection setup*

Configure the executable path and working directory, then launch the profiler. Nsight Compute generates a detailed profiling summary report:

![Figure 7.3: Nsight Compute Report Summary](/assets/posts/cuda-07-parallel-reduction/nc_summary.png)
*Figure 7.3: Nsight Compute execution summary table*

Under the summary tab, all kernel launches are listed with runtime duration, throughput, and compute utilization metrics. Double-clicking on a kernel opens the detailed metric breakdown with diagnostic recommendations:

![Figure 7.4: Nsight Compute Report Details](/assets/posts/cuda-07-parallel-reduction/nc_detail.png)
*Figure 7.4: Nsight Compute detailed workload analysis*

Clicking **Add Baseline** allows setting one profile as the reference point to directly inspect diffs against subsequent optimizations.

---

## 5. Optimization 1: Shared Memory Load Reduction [5]

In the baseline reduction kernel, the accumulation line:

```cpp
sdata[tid] += sdata[tid + s];
```

performs two shared memory loads and one shared memory store per iteration. Retaining `sdata[tid]` in a register accumulator avoids re-reading it from shared memory:

```cpp
    T sum = sdata[tid];
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] = sum = sum + sdata[tid + s];
        }
        __syncthreads();
    }
```

Measured performance:

| Version | Total Time (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |

In Nsight Compute's **Memory Workloads Analysis** section, comparing the shared memory statistics reveals a reduction of over 40% in `shared_load` instructions:

![Figure 7.5: Nsight Compute Shared Memory Access Statistics](/assets/posts/cuda-07-parallel-reduction/sm_table.png)
*Figure 7.5: Reduction in shared memory load instructions*

---

## 6. Optimization 2: Resolving Control Divergence [6]

A warp consists of 32 consecutive threads executed in lockstep. In `reduce_kernel_1` and `reduce_kernel_2`, the branching condition:

```cpp
if (tid % (2 * s) == 0)
```

causes severe branch divergence within warps. In the first stage ($s = 1$), odd and even threads diverge. The warp scheduler must execute both branches serially, disabling inactive threads via execution masks and cutting active execution throughput by half:

![Figure 7.6: Control Divergence Within a Warp](/assets/posts/cuda-07-parallel-reduction/control_divergence.png)
*Figure 7.6: Control divergence caused by modulo-based branch conditions*

Examining the access pattern:
* $s = 1$: thread $k$ processes elements `sdata[2k]` and `sdata[2k + 1]`.
* $s = 2$: thread $k$ processes elements `sdata[4k]` and `sdata[4k + 2]`.
* $s = 4$: thread $k$ processes elements `sdata[8k]` and `sdata[8k + 4]`.

The stage loop can be restructured by mapping consecutive thread IDs to the strided indices:

```cpp
    for (int s = 1; s < blockSize; s *= 2) {
        int idx = 2 * s * tid;
        if (idx < blockSize) {
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }
```

In this formulation, active threads are packed into contiguous thread IDs, preventing divergence across early warps until the number of active threads drops below 32:

| Version | Duration (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |
| `reduce_kernel_3` | 77.65 | 86.43 | 21.61 | 1.64 |

---

## 7. Optimization 3: Eliminating Bank Conflicts

Shared memory is physically partitioned into 32 equal-sized memory modules called banks:

![Figure 7.7: Conceptual Memory Model](/assets/posts/cuda-07-parallel-reduction/memory.png)
*Figure 7.7: Simplified memory cell grid*

Each bank occupies successive 32-bit (4-byte) words:

![Figure 7.8: 32-Bank Shared Memory Organization](/assets/posts/cuda-07-parallel-reduction/banks.png)
*Figure 7.8: Shared memory divided into 32 interleaved banks*

Access rules within the same warp:
1. Threads accessing distinct banks execute concurrently in a single clock cycle.
2. Multiple threads accessing the exact same address within a bank trigger a broadcast, serviced in a single cycle.
3. Multiple threads accessing different addresses within the same bank cause a bank conflict, forcing the hardware to serialize the conflicting requests.

In `reduce_kernel_3`, active threads compute `idx = 2 * s * tid`:
* When $s = 2$, thread 0 accesses `idx = 0` (bank 0) and thread 16 accesses `idx = 64` ($64 \pmod{32} = 0$, also bank 0).
* Threads 0 and 16 access different addresses within bank 0 simultaneously, resulting in a 2-way bank conflict:

![Figure 7.9: Bank Conflicts in Strided Shared Memory Access](/assets/posts/cuda-07-parallel-reduction/bank_conflicts.png)
*Figure 7.9: 2-way bank conflict between thread 0 and thread 16*

To eliminate bank conflicts, we reverse the loop direction to use sequential addressing with a decreasing stride $s$:

![Figure 7.10: Conflict-Free Sequential Addressing](/assets/posts/cuda-07-parallel-reduction/bank_conflicts_free.png)
*Figure 7.10: Sequential addressing eliminates bank conflicts*

In sequential addressing, active threads read consecutive memory locations:

```cpp
for (int s = blockSize / 2; s >= 1; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

Active threads (`tid < s`) access `sdata[tid]` and `sdata[tid + s]`. Since threads 0 through 31 access consecutive elements, requests map 1-to-1 to banks 0 through 31 without conflicts:

| Version | Duration (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |
| `reduce_kernel_3` | 77.65 | 86.43 | 21.61 | 1.64 |
| `reduce_kernel_4` | 60.37 | 111.16 | 27.79 | 2.11 |

Nsight Compute confirms that shared memory bank conflicts are reduced by over 99%:

![Figure 7.11: Nsight Compute Verification of Bank Conflict Reduction](/assets/posts/cuda-07-parallel-reduction/sm_table_bc.png)
*Figure 7.11: Elimination of bank conflicts in sequential reduction*

---

## 8. Optimization 4: Loop Unrolling

Loop control introduces overhead, including counter incrementation, conditional branch evaluation, and synchronization latencies. Since `blockSize` is a compile-time template parameter (maximum 1024 threads per block), the loop can be completely unrolled:

```cpp
    T sum = sdata[tid];
    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] = sum = sum + sdata[tid + 512];
        }
        __syncthreads();
    }
    
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] = sum = sum + sdata[tid + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] = sum = sum + sdata[tid + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] = sum = sum + sdata[tid + 64];
        }
        __syncthreads();
    }

    if (blockSize >= 64) {
        if (tid < 32) {
            sdata[tid] = sum = sum + sdata[tid + 32];
        }
        __syncthreads();
    }

    if (blockSize >= 32) {
        if (tid < 16) {
            sdata[tid] = sum = sum + sdata[tid + 16];
        }
        __syncthreads();
    }

    if (blockSize >= 16) {
        if (tid < 8) {
            sdata[tid] = sum = sum + sdata[tid + 8];
        }
        __syncthreads();
    }

    if (blockSize >= 8) {
        if (tid < 4) {
            sdata[tid] = sum = sum + sdata[tid + 4];
        }
        __syncthreads();
    }

    if (blockSize >= 4) {
        if (tid < 2) {
            sdata[tid] = sum = sum + sdata[tid + 2];
        }
        __syncthreads();
    }

    if (blockSize >= 2) {
        if (tid < 1) {
            sdata[tid] = sum = sum + sdata[tid + 1];
        }
        __syncthreads();
    }
```

Because `blockSize` is known at compile time, expressions like `if (blockSize >= 1024)` are evaluated by the compiler and branch dead code is eliminated at zero runtime cost. Alternatively, `#pragma unroll` can be applied to unroll loops automatically [7]:

```cpp
    T sum = sdata[tid];
    #pragma unroll
    for (int s = blockSize / 2; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sum = sum + sdata[tid + s];
        }
        __syncthreads();
    }
```

| Version | Duration (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |
| `reduce_kernel_3` | 77.65 | 86.43 | 21.61 | 1.64 |
| `reduce_kernel_4` | 60.37 | 111.16 | 27.79 | 2.11 |
| `reduce_kernel_5` | 59.90 | 112.03 | 28.01 | 2.13 |

---

## 9. Optimization 5: Unrolling the Final Warp

When the stride drops below 32 ($s \le 32$), all active threads belong to the same warp (warp 0). Under the classic SIMT execution model, instructions within a single warp were assumed to execute synchronously, allowing the removal of `__syncthreads()` and inner branch guards:

```cpp
template <typename T, unsigned int blockSize>
__device__ void warpReduce(volatile T* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}
```

The `volatile` qualifier signals to the compiler that `sdata` must not be cached into thread registers, guaranteeing that stores are written through to shared memory and immediately visible to peer threads.

> **Note**: While unrolling with `volatile` worked under the lockstep SIMD execution of pre-Volta architectures, GPUs starting with Volta (SM 7.0+) introduced Independent Thread Scheduling (ITS). On modern architectures, implicit lockstep execution is no longer guaranteed by the hardware. While converged code paths often produce correct results in simple benchmarks, NVIDIA officially classifies warp-synchronous shared memory access without explicit `__syncwarp()` barriers as an undefined race condition hazard. This architectural evolution motivated the industry-standard adoption of register-level warp shuffle instructions (`__shfl_down_sync`), covered in the next section.

| Version | Duration (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |
| `reduce_kernel_3` | 77.65 | 86.43 | 21.61 | 1.64 |
| `reduce_kernel_4` | 60.37 | 111.16 | 27.79 | 2.11 |
| `reduce_kernel_5` | 59.90 | 112.03 | 28.01 | 2.13 |
| `reduce_kernel_6` | 47.31 | 141.84 | 35.46 | 2.69 |

---

## 10. Optimization 6: Register Communication via Warp Shuffle

Communicating through shared memory incurs load/store instruction overhead. Starting with Kepler (and modernized in CUDA 9.0), NVIDIA introduced warp shuffle intrinsics (`__shfl_sync`), enabling threads within the same warp to exchange data directly through registers without shared memory transactions.

The `__shfl_down_sync` primitive shifts values across lanes within a warp:

```cpp
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);
```
* `mask`: A 32-bit bitmask indicating which threads participate in the shuffle (e.g., `0xffffffff` for all 32 lanes).
* `var`: The local register variable to read and transmit.
* `delta`: The lane offset from which to fetch data (current lane ID + `delta`).

![Figure 7.12: Data Movement with `__shfl_down_sync`](/assets/posts/cuda-07-parallel-reduction/shfl_down.png)
*Figure 7.12: Inter-lane data transfer using `__shfl_down_sync`*

Applying warp shuffle to the final warp reduction:

```cpp
template <typename T, unsigned int blockSize>
__device__ void warpShuffleSum(T &sum) {
    unsigned int FULL_MASK = 0xffffffff;
    if (blockSize >= 32) {
        sum += __shfl_down_sync(FULL_MASK, sum, 16);    
    }
    if (blockSize >= 16) {
        sum += __shfl_down_sync(FULL_MASK, sum, 8);
    }
    if (blockSize >= 8) {
        sum += __shfl_down_sync(FULL_MASK, sum, 4);
    }
    if (blockSize >= 4) {
        sum += __shfl_down_sync(FULL_MASK, sum, 2);
    }
    if (blockSize >= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, 1);
    }
}
```

Calling the shuffle function:

```cpp
if (blockSize >= 64) {
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();
}

if (tid < 32) {
    T sum = sdata[tid];
    warpShuffleSum<T, blockSize>(sum);
    if (tid == 0) {
        atomicAdd(&result[0], sum);
    }
}
```

Because `__shfl_down_sync` is a collective warp-level instruction requiring all specified lanes in the mask (`0xffffffff`) to execute concurrently, all 32 threads in warp 0 (`tid < 32`) must participate by loading their respective partial sums into registers before calling `warpShuffleSum`. After the intra-warp tree reduction converges, thread 0 writes the block sum to `result[0]` via `atomicAdd`.

| Version | Duration (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |
| `reduce_kernel_3` | 77.65 | 86.43 | 21.61 | 1.64 |
| `reduce_kernel_4` | 60.37 | 111.16 | 27.79 | 2.11 |
| `reduce_kernel_5` | 59.90 | 112.03 | 28.01 | 2.13 |
| `reduce_kernel_6` | 47.31 | 141.84 | 35.46 | 2.69 |
| `reduce_kernel_7` | 46.10 | 145.56 | 36.39 | 2.76 |

---

## 11. Optimization 7: Thread Coarsening

In preceding kernels, each thread initially loads a single element. In the first stage, only half the threads remain active; in subsequent stages, active thread count is halved repeatedly until only a single thread remains. Even though the majority of threads are idle during tree reduction, their thread block resources (registers and shared memory allocations) remain reserved on the SM until the entire block finishes execution.

Furthermore, assigning one thread per element requires creating thousands of blocks for large arrays. Because each SM can host a limited number of concurrent blocks, surplus blocks remain queued. Idle threads within running blocks cannot relinquish resources to waiting blocks, creating severe resource underutilization.

Thread coarsening resolves this mismatch by having each thread process multiple elements sequentially before entering the tree reduction:

![Figure 7.13: Thread Coarsening Resource Efficiency](/assets/posts/cuda-07-parallel-reduction/thread_coarsening.png)
*Figure 7.13: Comparison of single-element mapping vs. multi-element thread coarsening*

By configuring a fixed grid dimension and having threads iterate through global memory using a grid-stride loop, memory accesses from consecutive threads are coalesced:

![Figure 7.14: Grid-Stride Memory Access Pattern](/assets/posts/cuda-07-parallel-reduction/thread_load.png)
*Figure 7.14: Strided loading across global memory*

Implementation:

```cpp
    __shared__ T sdata[blockSize];

    int gridSize = blockSize * gridDim.x;
    int tid = threadIdx.x;
    int globalId = tid + blockIdx.x * blockSize;

    sdata[tid] = 0;
    T sum = sdata[tid];

    while (globalId < n) {
        sum += data[globalId];
        globalId += gridSize;
    }

    sdata[tid] = sum;
    __syncthreads();
```

| Version | Duration (ms) | Throughput (GB/s) | Performance (GFLOPS) | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| `reduce_kernel_1` | 127.29 | 52.72 | 13.18 | 1.00 |
| `reduce_kernel_2` | 123.18 | 54.48 | 13.62 | 1.03 |
| `reduce_kernel_3` | 77.65 | 86.43 | 21.61 | 1.64 |
| `reduce_kernel_4` | 60.37 | 111.16 | 27.79 | 2.11 |
| `reduce_kernel_5` | 59.90 | 112.03 | 28.01 | 2.13 |
| `reduce_kernel_6` | 47.31 | 141.84 | 35.46 | 2.69 |
| `reduce_kernel_7` | 46.10 | 145.56 | 36.39 | 2.76 |
| `reduce_kernel_8` | 37.09 | 180.92 | 45.23 | 3.43 |

With thread coarsening, memory throughput reaches $180.92\text{ GB/s}$ ($180.92 / 192 = 94.23\%$ of hardware theoretical bandwidth), approaching the Roofline ceiling of 48 GFLOPS.

---

## 12. Summary

Through the iterative optimization of parallel reduction, we explored fundamental CUDA performance optimization techniques: resolving control divergence, eliminating bank conflicts, unrolling loops, utilizing register-level warp shuffle intrinsics, and employing thread coarsening. Full source code is available on [GitLab](https://gitlab.com/cuda_exercise/reduce).

---

## References

<ul class="list-none pl-0 space-y-3">
  <li id="ref-0">
    Mark Harris, <a href="https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf" target="_blank" rel="noopener noreferrer">Optimizing Parallel Reduction in CUDA</a>, NVIDIA Corporation.
  </li>
  <li id="ref-1">
    TechPowerUp, <a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4050-mobile.c3953" target="_blank" rel="noopener noreferrer">NVIDIA GeForce RTX 4050 Mobile Specs</a>.
  </li>
  <li id="ref-2">
    Atomic operations vary by architecture; Ampere and newer architectures feature high-speed hardware-accelerated integer and FP32 atomics in L2 cache, whereas older architectures or unsupported types serialize operations.
  </li>
  <li id="ref-3">
    Reza Zadeh, <a href="https://stanford.edu/~rezab/dao/notes/lecture01/cme323_lec1.pdf" target="_blank" rel="noopener noreferrer">CME 323: Distributed Algorithms and Optimization</a>, Stanford University.
  </li>
  <li id="ref-4">
    NVIDIA Corporation, <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html" target="_blank" rel="noopener noreferrer">Kernel Profiling Guide</a>, Nsight Compute Documentation.
  </li>
  <li id="ref-5">
    NVIDIA Corporation, <a href="https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu#L196" target="_blank" rel="noopener noreferrer">reduction_kernel.cu</a>, CUDA Samples Repository.
  </li>
  <li id="ref-6">
    David B. Kirk and Wen-mei W. Hwu, <em>Programming Massively Parallel Processors: A Hands-on Approach</em> (PMPP), Chapter 4.5: Control Divergence.
  </li>
  <li id="ref-7">
    In modern compilers, manual loop unrolling can occasionally hinder compiler scheduling heuristics; `#pragma unroll` is typically preferred.
  </li>
</ul>
