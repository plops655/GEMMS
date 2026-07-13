#pragma once

#include <iostream>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

inline constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

// Fast exp via PTX exp2 approximation: exp(x) = 2^(x * log2(e))
__device__ __forceinline__
float ptx_exp2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

__device__ __forceinline__
float fast_exp(float x) {
  return ptx_exp2(x * 1.4427f);  // 1.4427 ≈ log2(e)
}

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// ============================================================================
// FP8 helpers
// ============================================================================

// Convert a single FP8 E4M3 value to BF16 in registers
__device__ __forceinline__
nv_bfloat16 fp8e4m3_to_bf16(__nv_fp8_e4m3 val) {
  return static_cast<nv_bfloat16>(static_cast<float>(val));
}

// Convert 4 packed FP8 values (in a uint32_t) to 4 BF16 values (2 × uint32_t)
// Input:  fp8x4 = [e0, e1, e2, e3] packed as 4 bytes in one register
// Output: bf16x2[0] = {bf16(e0), bf16(e1)}, bf16x2[1] = {bf16(e2), bf16(e3)}
__device__ __forceinline__
void fp8x4_to_bf16x2(uint32_t fp8x4, uint32_t bf16x2[2]) {
  // Unpack 4 FP8 bytes → 4 individual floats → repack as 2 pairs of BF16
  const __nv_fp8_e4m3 *p = reinterpret_cast<const __nv_fp8_e4m3*>(&fp8x4);
  nv_bfloat162 pair0 = __floats2bfloat162_rn(static_cast<float>(p[0]),
                                              static_cast<float>(p[1]));
  nv_bfloat162 pair1 = __floats2bfloat162_rn(static_cast<float>(p[2]),
                                              static_cast<float>(p[3]));
  bf16x2[0] = *reinterpret_cast<uint32_t*>(&pair0);
  bf16x2[1] = *reinterpret_cast<uint32_t*>(&pair1);
}

// Native FP8 E4M3 tensor-core MMA: m16 × n8 × k32
// A: row-major [16, 32] FP8 E4M3 — 4 registers (each holds 8 FP8 values)
// B: col-major [8, 32]  FP8 E4M3 — 2 registers (each holds 8 FP8 values)
// D: FP32 accumulator — 4 registers
// Uses the Blackwell / Hopper native FP8 MMA path for 2× K throughput vs BF16
__device__ __forceinline__
void mma_m16n8k32_fp8(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13};"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3])
  );
}

// Async global→shared copy for FP8 data (16 bytes = 16 FP8 elements per cp.async)
// HEIGHT × WIDTH in FP8 elements, TB_SIZE = threads per block
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_fp8(uint32_t dst, const __nv_fp8_e4m3 *src, int src_stride, int tid) {
  constexpr int NUM_ELEMS = 16;  // 16 bytes = 16 FP8 values per async copy
  constexpr int TOTAL_COPIES = HEIGHT * WIDTH / NUM_ELEMS;
  constexpr int ITERS = TOTAL_COPIES / TB_SIZE;
  constexpr int REMAINDER = TOTAL_COPIES % TB_SIZE;

  #pragma unroll
  for (int iter = 0; iter < ITERS; iter++) {
    const int copy_id = iter * TB_SIZE + tid;
    const int elem_idx = copy_id * NUM_ELEMS;
    const int row = elem_idx / WIDTH;
    const int col = elem_idx % WIDTH;
    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(__nv_fp8_e4m3);
    const __nv_fp8_e4m3 *src_addr = src + row * src_stride + col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
  // Handle remainder iterations (if TOTAL_COPIES not divisible by TB_SIZE)
  if (REMAINDER > 0) {
    const int copy_id = ITERS * TB_SIZE + tid;
    if (copy_id < TOTAL_COPIES) {
      const int elem_idx = copy_id * NUM_ELEMS;
      const int row = elem_idx / WIDTH;
      const int col = elem_idx % WIDTH;
      const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(__nv_fp8_e4m3);
      const __nv_fp8_e4m3 *src_addr = src + row * src_stride + col;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    }
  }
}

// Swizzled version of FP8 async copy (avoids shared-memory bank conflicts)
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_fp8_swizzle(uint32_t dst, const __nv_fp8_e4m3 *src, int src_stride, int tid) {
  constexpr int NUM_ELEMS = 16;
  constexpr int TOTAL_COPIES = HEIGHT * WIDTH / NUM_ELEMS;
  constexpr int ITERS = TOTAL_COPIES / TB_SIZE;
  constexpr int REMAINDER = TOTAL_COPIES % TB_SIZE;

  #pragma unroll
  for (int iter = 0; iter < ITERS; iter++) {
    const int copy_id = iter * TB_SIZE + tid;
    const int elem_idx = copy_id * NUM_ELEMS;
    const int row = elem_idx / WIDTH;
    const int col = elem_idx % WIDTH;
    // Swizzle destination address to avoid bank conflicts when threads in a warp
    // access the same shared-memory bank. XOR-based swizzle permutes the lower
    // address bits so that consecutive rows map to different banks.
    const uint32_t dst_addr = swizzle<WIDTH * sizeof(__nv_fp8_e4m3)>(
        dst + (row * WIDTH + col) * sizeof(__nv_fp8_e4m3));
    const __nv_fp8_e4m3 *src_addr = src + row * src_stride + col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
  if (REMAINDER > 0) {
    const int copy_id = ITERS * TB_SIZE + tid;
    if (copy_id < TOTAL_COPIES) {
      const int elem_idx = copy_id * NUM_ELEMS;
      const int row = elem_idx / WIDTH;
      const int col = elem_idx % WIDTH;
      const uint32_t dst_addr = swizzle<WIDTH * sizeof(__nv_fp8_e4m3)>(
          dst + (row * WIDTH + col) * sizeof(__nv_fp8_e4m3));
      const __nv_fp8_e4m3 *src_addr = src + row * src_stride + col;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    }
  }
}

// Commit pending cp.async copies and wait for all to land in shared memory
__device__ __forceinline__
void cp_async_commit_and_wait() {
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
}

// Commit group and wait for all but N groups (for double-buffering: wait_group 1)
template <int N>
__device__ __forceinline__
void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N));
  __syncthreads();
}

__device__ __forceinline__
void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <typename T, typename... Args>
void launch_kernel(
  T *kernel,
  int num_blocks,
  int block_size,
  int smem_size,
  Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<num_blocks, block_size, smem_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}