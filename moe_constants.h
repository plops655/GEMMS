#pragma once

// ============================================================================
// DeepSeek-V3 / R1 — Fixed Model Geometry
// ============================================================================

inline constexpr int H_DIM          = 7168;   // hidden size
inline constexpr int I_DIM          = 2048;   // intermediate size (per-expert FFN)
inline constexpr int E_GLOBAL       = 256;    // total routed experts
inline constexpr int E_LOCAL        = 32;     // experts on this rank (expert-parallel)

// ============================================================================
// Routing constants
// ============================================================================

inline constexpr int TOP_K          = 8;      // experts selected per token
inline constexpr int N_GROUP        = 8;      // number of expert groups
inline constexpr int TOPK_GROUP     = 4;      // groups kept after group-level top-k
inline constexpr int GROUP_SIZE     = E_GLOBAL / N_GROUP;  // 32 experts per group

// ============================================================================
// FP8 block-quantization granularity
// ============================================================================

inline constexpr int BLOCK_QUANT    = 128;    // elements per quantization block

// Derived block counts
inline constexpr int NUM_H_BLOCKS   = H_DIM / BLOCK_QUANT;          // 56
inline constexpr int NUM_I_BLOCKS   = I_DIM / BLOCK_QUANT;          // 16
inline constexpr int NUM_2I_BLOCKS  = (2 * I_DIM) / BLOCK_QUANT;    // 32

// ============================================================================
// Kernel tuning — persistent megakernel (target: B200, 192 SMs)
// ============================================================================

// Thread-block geometry
inline constexpr int NUM_WARPS          = 4;
inline constexpr int THREADS_PER_BLOCK  = NUM_WARPS * 32;  // 128

// Number of persistent blocks — one per SM on B200
inline constexpr int NUM_SMS            = 192;

// GEMM tiling (all GEMMs use the same K-tile = BLOCK_QUANT for scale alignment)
inline constexpr int K_TILE             = BLOCK_QUANT;  // 128
inline constexpr int N_TILE             = 128;          // output-column tile

// GEMM1: [Tk, H] × [H, 2I] — H is the K-dimension
//   Number of K-tiles = H / K_TILE = 56
//   Number of N-tile pairs = I / N_TILE = 16
//   (each "pair" computes gate[j] and up[j] columns, then fuses SwiGLU)
inline constexpr int GEMM1_K_TILES      = H_DIM / K_TILE;         // 56
inline constexpr int GEMM1_N_PAIRS      = I_DIM / N_TILE;         // 16
inline constexpr int GEMM1_WORK_UNITS   = E_LOCAL * GEMM1_N_PAIRS;  // 512

// GEMM2: [Tk, I] × [I, H] — I is the K-dimension
//   Number of K-tiles = I / K_TILE = 16
//   Number of N-tiles = H / N_TILE = 56
inline constexpr int GEMM2_K_TILES      = I_DIM / K_TILE;         // 16
inline constexpr int GEMM2_N_TILES      = H_DIM / N_TILE;         // 56
inline constexpr int GEMM2_WORK_UNITS   = E_LOCAL * GEMM2_N_TILES;  // 1792

// M-dimension tile for tcgen05 warpgroup MMA (fixed: m=128 for cta_group::1)
// NOTE: With small T (e.g. T=64, ~2 tokens/expert), most M-rows are padding.
//       At T≥512 (continuous-batching scenarios), M_TILE=128 approaches full efficiency.
inline constexpr int M_TILE             = 128;

// K-dimension per tcgen05 MMA call (FP8 kind::f8f6f4 → K=64 per step)
// GEMM1 K-tile is still K_TILE=128 bytes from DRAM; each K-tile = 2 MMA steps of K=64.
inline constexpr int TG05_K_STEP        = 64;

// ============================================================================
// Shared-memory budget (B200: 228 KB per SM, tcgen05 path)
//
// Phase 1 (GEMM1 + SwiGLU):
//   A_buf:  2 × M_TILE × K_TILE × 1B (FP8, double-buffered)     =  32 KB
//   B_buf:  2 × N_TILE × K_TILE × 1B (FP8, double-buffered)     =  32 KB
//   mbar:   2 × 8B (TMA completion barriers, one per buffer)     = negligible
//   gate_acc: REMOVED — accumulator lives in TMEM now            =   0 KB
//                                                         Total  =  64 KB
//
// Phase 2 (GEMM2 + accumulate):
//   A_buf:  2 × M_TILE × K_TILE × 2B (BF16, double-buffered)    =  64 KB
//   B_buf:  2 × N_TILE × K_TILE × 1B (FP8, double-buffered)     =  32 KB
//   mbar:   2 × 8B                                               = negligible
//                                                         Total  =  96 KB
//
// KernelSMEM union = max(64, 96) = 96 KB → floor(228/96) = 2 blocks/SM
// TMEM (separate from SMEM): 128 cols × 512B/col = 64 KB (gate or up at a time)
// ============================================================================

inline constexpr int SMEM_PHASE1 = 2 * M_TILE * K_TILE * 1   // A double-buf FP8
                                 + 2 * N_TILE * K_TILE * 1;  // B double-buf FP8
                                                              // = 64 KB

inline constexpr int SMEM_PHASE2 = 2 * M_TILE * K_TILE * 2   // A double-buf BF16
                                 + 2 * N_TILE * K_TILE * 1;  // B double-buf FP8
                                                              // = 96 KB

// Use the larger of the two (they share SMEM via union)
inline constexpr int SMEM_TOTAL  = (SMEM_PHASE1 > SMEM_PHASE2) ? SMEM_PHASE1 : SMEM_PHASE2;
