// ============================================================================
// blackwell.h — B200 (sm_100) Native Instruction Library
//
// Data flow:
//
//  DRAM ──[TMA]──► SMEM (swizzled) ──[tcgen05.mma]──► TMEM (accumulator)
//                                                           │
//                                                      [tcgen05.ld]
//                                                           │
//                                                      Registers
//                                                           │
//                                                      Store to DRAM
//
// Three subsystems:
//
//  1. TMA — single-thread bulk async DMA, hardware swizzles SMEM writes
//  2. mbar — memory barrier in SMEM, signals TMA completion to all threads
//  3. tcgen05 — warpgroup MMA (M=128,N=128,K=64 FP8), accumulates into TMEM
//
// Why each replaces Ampere:
//
//  cp.async (Ampere): 128 threads each issue 16B loads in a loop
//  TMA       (sm100): 1 thread issues the entire [M×K] tile, hardware does it
//
//  cp.async.wait_group + __syncthreads (Ampere): counting-based, coarse
//  mbarrier              (sm100): byte-counting, signals exactly when tile lands
//
//  mma.sync m16n8k32 (Ampere): 1 warp, accumulator in registers (~32 regs/warp)
//  tcgen05 m128n128k64(sm100): 4 warps (1 warpgroup), accumulator in TMEM (0 regs)
// ============================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// tcgen05 tile shape (fixed for cta_group::1, kind::f8f6f4)
// ============================================================================

inline constexpr int TG05_M    = 128;  // M rows per MMA tile (1 warpgroup = 128 threads)
inline constexpr int TG05_N    = 128;  // N cols per MMA tile
inline constexpr int TG05_K    = 64;   // K width per MMA call (FP8 → K=64)

// TMEM column count for M=128, N=128, FP32 accumulator.
// Each TMEM column = 128 threads × 4B = 512B (one float per thread).
// N=128 output columns → 128 TMEM columns.
inline constexpr int TMEM_COLS = TG05_N;

// ============================================================================
// Part 1: TMA descriptor creation — HOST SIDE
// ============================================================================
//
// The TMA descriptor (CUtensorMap, 128 bytes) encodes EVERYTHING the hardware
// DMA engine needs: global base pointer, tensor shape, row strides, tile size,
// element type, and swizzle mode.
//
// Key parameters:
//   CU_TENSOR_MAP_SWIZZLE_128B — hardware XORs column addresses with
//     ((row % 8) * 16) during the write to SMEM. This distributes 8
//     consecutive rows across all 32 SMEM banks, eliminating the bank
//     conflict that occurs when multiple warps' ldmatrix reads hit the
//     same bank on the same row.
//
//   CU_TENSOR_MAP_L2_PROMOTION_L2_128B — hints L2 to speculatively
//     fetch the next 128-byte cache line, improving DRAM→L2 efficiency.
//
// Coordinates passed at load time (device side) select which box to copy.
// For a [num_rows × num_cols] tensor with [tile_rows × tile_cols] boxes:
//   coord_col = k_tile * tile_cols   (column offset in elements)
//   coord_row = token_offset         (row offset in elements)

#ifndef __CUDACC_RTC__
#include <cuda.h>   // CUtensorMap, cuTensorMapEncodeTiled

// FP8 tensor [num_rows, num_cols] — loads [tile_rows × tile_cols] boxes
inline void create_tma_fp8(
    CUtensorMap*  desc,
    const void*   global_ptr,
    uint64_t      num_rows,
    uint64_t      num_cols,
    uint32_t      tile_rows,
    uint32_t      tile_cols
) {
    // TMA uses column-major dimension order: innermost (fastest) dimension first.
    uint64_t globalDim[2]    = { num_cols, num_rows };
    uint64_t globalStride[1] = { num_cols * sizeof(__nv_fp8_e4m3) }; // bytes/row
    uint32_t boxDim[2]       = { tile_cols, tile_rows };              // innermost first
    uint32_t elemStride[2]   = { 1, 1 };                              // dense layout

    CUresult r = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,      // FP8 → uint8 for addressing
        2,                                   // 2D tensor
        (void*)global_ptr,
        globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,          // ← swizzle writes to SMEM
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,  // ← hint L2 prefetch 128B lines
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (r != CUDA_SUCCESS) {
        const char* s; cuGetErrorName(r, &s);
        fprintf(stderr, "[TMA] create_tma_fp8 failed: %s\n", s);
    }
}

// BF16 tensor [num_rows, num_cols]
inline void create_tma_bf16(
    CUtensorMap*  desc,
    const void*   global_ptr,
    uint64_t      num_rows,
    uint64_t      num_cols,
    uint32_t      tile_rows,
    uint32_t      tile_cols
) {
    uint64_t globalDim[2]    = { num_cols, num_rows };
    uint64_t globalStride[1] = { num_cols * sizeof(nv_bfloat16) };
    uint32_t boxDim[2]       = { tile_cols, tile_rows };
    uint32_t elemStride[2]   = { 1, 1 };

    cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)global_ptr,
        globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}
#endif  // !__CUDACC_RTC__

// ============================================================================
// Part 2: Memory Barrier (mbar) — DEVICE SIDE
// ============================================================================
//
// Protocol for each double-buffered iteration:
//
//  Thread 0 only:
//    mbar_init(mbar[buf], 1)          ← "expect 1 arrive_tx call"
//    mbar_arrive_expect_tx(mbar[buf], N_bytes)  ← "N bytes are incoming"
//    tma_load_2d(smem_dst, tma_desc, col, row, &mbar[buf])  ← issue DMA
//
//  ALL threads:
//    mbar_wait(&mbar[buf], parity)    ← spin until N_bytes have arrived
//    // SMEM tile is now safe to read
//
// The parity bit alternates 0→1→0→... each time the mbar completes.
// This lets you REUSE the same mbar for successive double-buffer iterations
// without reinitializing it, by passing the correct parity each time.
//
// Why mbar instead of cp_async_wait_group?
//   cp_async_wait_group counts "groups" (commit calls), not bytes.
//   It can only guarantee all copies in a group landed — not WHEN.
//   mbar counts exact byte transfers, so it triggers the moment the last
//   byte of the tile is written to SMEM. Zero wasted wait cycles.

__device__ __forceinline__
void mbar_init(uint64_t* mbar, uint32_t expected_arrives = 1) {
    // Sets the mbar to "open" after 'expected_arrives' arrive_tx calls.
    // For TMA: always 1 (thread 0 calls arrive_tx once per tile).
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;"
        :: "r"(addr), "r"(expected_arrives) : "memory"
    );
}

__device__ __forceinline__
void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t expected_bytes) {
    // Called by thread 0 BEFORE issuing TMA load.
    // Tells the mbar: "I expect expected_bytes of async data to arrive."
    // The TMA engine atomically decrements this count as bytes land in SMEM.
    // When count reaches 0, the mbar flips its parity and wakes all waiters.
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(expected_bytes) : "memory"
    );
}

__device__ __forceinline__
void mbar_wait(uint64_t* mbar, uint32_t parity) {
    // All threads call this. Spins until mbar's phase matches 'parity'.
    // Uses try_wait (non-blocking poll) in a loop.
    // The 'fence.proxy.async' ensures async copies are visible before reading.
    asm volatile("fence.proxy.async;" ::: "memory");
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    #pragma unroll 1
    while (true) {
        uint32_t done;
        asm volatile(
            "{\n"
            " .reg .pred p;\n"
            " mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done) : "r"(addr), "r"(parity)
        );
        if (done) break;
    }
}

// ============================================================================
// Part 3: TMA Load — DEVICE SIDE
// ============================================================================
//
// Loads a 2D tile from DRAM → SMEM. Issued by ONE thread only.
// Hardware writes to SMEM with 128B swizzle applied automatically.
//
//  Old way (cp.async): 128 threads × loop × 16B = 128 ptx instructions issued
//  TMA way:            1 thread issues 1 ptx instruction, hardware does the rest
//
// Args:
//   smem_dst   — uint32_t SMEM address (from __cvta_generic_to_shared)
//   tma_desc   — pointer to the 128-byte CUtensorMap on device (__grid_constant__)
//   coord_col  — column start in the GLOBAL tensor (element index, not byte)
//   coord_row  — row start in the GLOBAL tensor (element index)
//   mbar       — the mbar in SMEM that TMA will signal on completion

__device__ __forceinline__
void tma_load_2d(
    uint32_t            smem_dst,
    const CUtensorMap*  tma_desc,   // __grid_constant__ in kernel param list
    int                 coord_col,
    int                 coord_row,
    uint64_t*           mbar
) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst),
           "l"(tma_desc),      // 'l' constraint = 64-bit pointer
           "r"(coord_col),
           "r"(coord_row),
           "r"(mbar_addr)
        : "memory"
    );
}

// Convenience: compute SMEM address of a buffer given SMEM base + byte offset
__device__ __forceinline__
uint32_t smem_addr_of(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

// ============================================================================
// Part 4: SMEM Descriptor for tcgen05 MMA
// ============================================================================
//
// tcgen05.mma reads SMEM operands via 64-bit DESCRIPTORS — not raw pointers.
// The descriptor tells the tensor core engine WHERE in SMEM the tile is
// and HOW it is laid out (including swizzle).
//
// Bit layout of the 64-bit descriptor:
//   [13:0]  = smem_base_addr >> 4     (16-byte aligned, 14-bit field)
//   [29:16] = row_stride_bytes >> 4   (leading dimension in 16B units)
//   [63:62] = swizzle mode            (3 = 128B, 2 = 64B, 1 = 32B, 0 = none)
//
// The swizzle bits MUST match what TMA used to write the tile.
// The MMA engine uses the same formula to de-swizzle on reads,
// so the right values reach the tensor core inputs regardless of layout.
//
// Example: FP8 tile [128 × 128], stride = 128 bytes
//   desc = (smem_ptr >> 4)                   // [13:0]
//        | ((128 >> 4) << 16)                // [29:16] = 8 (=128/16 units)
//        | (3ULL << 62)                      // [63:62] = 128B swizzle

__device__ __forceinline__
uint64_t make_smem_desc(const void* smem_ptr, uint32_t row_stride_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>(addr >> 4) & 0x3FFF;               // [13:0]
    desc |= static_cast<uint64_t>(row_stride_bytes >> 4 & 0x3FFF) << 16; // [29:16]
    desc |= static_cast<uint64_t>(3) << 62;  // [63:62] = 128B swizzle
    return desc;
}

// ============================================================================
// Part 5: TMEM — Tensor Memory management
// ============================================================================
//
// TMEM is a B200-only on-chip memory bank dedicated to MMA accumulators.
//
// Memory hierarchy on B200 (fastest → slowest):
//   Registers  (255 × 4B per thread = 130 KB per SM at 128 threads/block)
//   TMEM       (dedicated MMA accumulator bank, 0 register cost)
//   SMEM       (228 KB per SM, shared among all blocks on the SM)
//   L2 cache   (256 MB, shared GPU-wide)
//   HBM        (288 GB/s bandwidth)
//
// For M=128, N=128, FP32 accumulator:
//   128 × 128 × 4B = 64 KB needed
//   TMEM columns: 128 (one per N output value)
//   Each column = 128 threads × 4B = 512B
//
// With old mma.sync (Ampere):   accumulator held in registers
//   4 warps × 32 regs/warp = 128 regs consumed JUST for the accumulator
// With tcgen05 (Blackwell):      accumulator in TMEM
//   0 registers used for accumulator → free up ~128 regs for other uses
//   → higher occupancy, less register spill
//
// IMPORTANT: tcgen05.alloc must be called by ALL threads simultaneously.
// The returned tmem_addr is the same for all threads in the CTA.

// N_COLS must be a compile-time constant — PTX requires an immediate operand.
template <uint32_t N_COLS>
__device__ __forceinline__
uint32_t tcgen05_alloc() {
    uint32_t tmem_addr;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned %0, %1;"
        : "=r"(tmem_addr) : "n"(N_COLS)   // "n" = compile-time immediate
    );
    return tmem_addr;
}

template <uint32_t N_COLS>
__device__ __forceinline__
void tcgen05_dealloc(uint32_t tmem_addr) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned %0, %1;"
        :: "r"(tmem_addr), "n"(N_COLS)    // "n" = compile-time immediate
    );
}

// ============================================================================
// Part 6: tcgen05 MMA — warpgroup matrix multiply-accumulate
// ============================================================================
//
// tcgen05.mma.cta_group::1.kind::f8f6f4
//   Computes: TMEM[m,n] += A_smem[m,k] × B_smem[n,k]   (A row-major, B col-major)
//   Tile:     M=128, N=128, K=64 (FP8 inputs, FP32 accumulation)
//
// ALL 128 threads (4 warps) issue this instruction — it is a WARPGROUP operation.
// The 4 warps divide the 128 M-rows among themselves:
//   Warp 0 (threads   0-31):  M-rows   0-31
//   Warp 1 (threads  32-63):  M-rows  32-63
//   Warp 2 (threads  64-95):  M-rows  64-95
//   Warp 3 (threads 96-127):  M-rows 96-127
//
// 'accumulate':
//   false → D = A × B          (initialize accumulator — first K-tile)
//   true  → D = A × B + C      (accumulate — subsequent K-tiles)
//
// a_desc: SMEM descriptor for A tile [M=128, K=64] in SMEM (from make_smem_desc)
// b_desc: SMEM descriptor for B tile [N=128, K=64] in SMEM (B is transposed)

__device__ __forceinline__
void tcgen05_mma_fp8(
    uint32_t tmem_addr,
    uint64_t a_desc,
    uint64_t b_desc,
    bool accumulate
) {
    // scale_d (accumulate flag) MUST be a literal 0 or 1 in PTX — not a register.
    if (accumulate) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, 1;"
            :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc) : "memory"
        );
    } else {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, 0;"
            :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc) : "memory"
        );
    }
}

__device__ __forceinline__
void tcgen05_commit(uint32_t tmem_addr) {
    // Commit all pending MMA operations to TMEM.
    // After this, the accumulator is stable and can be read.
    asm volatile(
        "tcgen05.commit.cta_group::1 [%0];"
        :: "r"(tmem_addr) : "memory"
    );
}

__device__ __forceinline__
void tcgen05_fence() {
    // Memory fence: ensures all threads see committed TMEM state.
    // Must be called after tcgen05_commit(), before tcgen05_ld().
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ============================================================================
// Part 7: TMEM Readback — tcgen05.ld
// ============================================================================
//
// After commit + fence, reads the FP32 accumulator from TMEM to registers.
//
// TMEM layout for M=128, N=128, FP32:
//   Each TMEM "column" c (0..127) holds one N-output value per thread.
//   Thread t (0..127) holds M-row t.
//   The float at output[row=t][col=c] is at:
//     byte offset within TMEM = c × (THREADS_PER_BLOCK × sizeof(float))
//                             = c × 512
//
// tcgen05.ld.b32x4 reads 4 consecutive columns in one instruction:
//   out[0] = output[m=tid][n=nc+0]
//   out[1] = output[m=tid][n=nc+1]
//   out[2] = output[m=tid][n=nc+2]
//   out[3] = output[m=tid][n=nc+3]
//
// To read all N=128 values: call tcgen05_ld_4 for nc = 0, 4, 8, ..., 124
// (32 calls per thread, 4 N-values each = 128 total N-values).

__device__ __forceinline__
void tcgen05_ld_4(
    uint32_t  out[4],
    uint32_t  tmem_addr,
    uint32_t  col_byte_offset  // = col_index × THREADS_PER_BLOCK × 4
) {
    uint32_t addr = tmem_addr + col_byte_offset;
    asm volatile(
        "tcgen05.ld.sync.aligned.4x32b {%0, %1, %2, %3}, [%4];"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(addr)
        : "memory"
    );
}

// ============================================================================
// Inline double-buffer TMA helper — loads A + B tiles in one shot
// ============================================================================
//
// Issued by thread 0, waits by all threads.
// Returns the parity bit to use for the NEXT call to mbar_wait() on this mbar.
//
// Usage:
//   // Before loop: prefetch tile 0 → buf 0
//   tma_load_ab(smem_A[0], tma_A, k*K, token_row,
//               smem_B[0], tma_B, n*N, k*K,
//               &mbar[0], tile_bytes, 0, tid);
//   mbar_wait(&mbar[0], /*parity=*/0);
//
//   for kt = 0..K_TILES-1:
//     buf = kt & 1
//     // prefetch tile kt+1 into buf 1-buf
//     if (kt+1 < K_TILES):
//       tma_load_ab(..., &mbar[1-buf], tile_bytes, kt+1, tid)
//     // compute MMA on buf
//     tcgen05_mma_fp8(tmem, desc_A[buf], desc_B[buf], kt > 0)
//     // wait for next tile (issued above)
//     if (kt+1 < K_TILES):
//       mbar_wait(&mbar[1-buf], parity for buf 1-buf)

__device__ __forceinline__
void tma_prefetch_ab(
    uint32_t            smem_A,       // SMEM dst for A tile
    const CUtensorMap*  tma_A,
    int                 a_col, int a_row,
    uint32_t            smem_B,       // SMEM dst for B tile
    const CUtensorMap*  tma_B,
    int                 b_col, int b_row,
    uint64_t*           mbar,
    uint32_t            total_bytes,  // A bytes + B bytes
    int                 tid
) {
    if (tid == 0) {
        mbar_arrive_expect_tx(mbar, total_bytes);
        tma_load_2d(smem_A, tma_A, a_col, a_row, mbar);
        tma_load_2d(smem_B, tma_B, b_col, b_row, mbar);
    }
    // Callers must call mbar_wait() after this.
}
