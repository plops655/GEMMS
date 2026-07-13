// ============================================================================
// Persistent MoE Megakernel v2 — DeepSeek-V3 / R1
//
// Fixes over v1:
//   1. Routing split into separate kernel (isolates register spill from GEMMs)
//   2. Permutation build parallelized with atomics (no serial Block 0 bottleneck)
//   3. Regular launch (no cooperative groups / grid.sync())
//   4. Phase 2 work-skipping instead of spin-waiting
//
// Architecture:
//   Kernel 1a: routing_topk_kernel — compute top-k routing per token
//   Kernel 1b: routing_permute_kernel — build sorted_token_ids with atomics
//   Kernel 2:  persistent_gemm_kernel — Phase 1 (GEMM1+SwiGLU) + Phase 2 (GEMM2+accum)
//
// Target: NVIDIA B200 (192 SMs, 228 KB SMEM, native FP8 tensor cores)
// ============================================================================

#include "common.h"
#include "moe_constants.h"
#include "blackwell.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

// ============================================================================
// Scratch layout v2
// ============================================================================

struct ScratchOffsets2 {
  size_t topk_idx;              // [T, TOP_K] int
  size_t topk_weights;          // [T, TOP_K] float
  size_t sorted_token_ids;      // [E_LOCAL, T] int
  size_t expert_token_count;    // [E_LOCAL] int
  size_t expert_token_slot;     // [E_LOCAL] int — atomic scatter counters
  size_t gemm1_work_counter;    // int
  size_t gemm2_work_counter;    // int
  size_t gemm2_done_counter;    // int — counts completed Phase 2 units
  size_t gemm1_done;            // [E_LOCAL] int
  size_t gemm2_claimed;         // [GEMM2_WORK_UNITS] int — claim flags
  size_t intermediate;          // [E_LOCAL, T, I_DIM] BF16
  size_t output;                // [T, H_DIM] FP32
  size_t total_bytes;
};

inline ScratchOffsets2 compute_scratch_offsets2(int T) {
  ScratchOffsets2 o;
  size_t cur = 0;
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) / a * a; };

  o.topk_idx           = cur; cur += align_up(T * TOP_K * sizeof(int), 128);
  o.topk_weights       = cur; cur += align_up(T * TOP_K * sizeof(float), 128);
  o.sorted_token_ids   = cur; cur += align_up(E_LOCAL * T * sizeof(int), 128);
  o.expert_token_count = cur; cur += align_up(E_LOCAL * sizeof(int), 128);
  o.expert_token_slot  = cur; cur += align_up(E_LOCAL * sizeof(int), 128);
  o.gemm1_work_counter = cur; cur += align_up(sizeof(int), 128);
  o.gemm2_work_counter = cur; cur += align_up(sizeof(int), 128);
  o.gemm2_done_counter = cur; cur += align_up(sizeof(int), 128);
  o.gemm1_done         = cur; cur += align_up(E_LOCAL * sizeof(int), 128);
  o.gemm2_claimed      = cur; cur += align_up(GEMM2_WORK_UNITS * sizeof(int), 128);
  o.intermediate       = cur; cur += align_up((size_t)E_LOCAL * T * I_DIM * sizeof(nv_bfloat16), 128);
  o.output             = cur; cur += align_up((size_t)T * H_DIM * sizeof(float), 128);
  o.total_bytes        = cur;
  return o;
}

// ============================================================================
// Device helpers
// ============================================================================

template <typename T>
__host__ __device__ __forceinline__
T* scratch_ptr(char* scratch, size_t byte_offset) {
  return reinterpret_cast<T*>(scratch + byte_offset);
}

// ============================================================================
// Prefetch helpers — used by the persistent GEMM kernel's double-buffered loops
// ============================================================================

// Prefetch FP8 A tile (scattered token rows → SMEM)
__device__ __forceinline__
void prefetch_A_fp8(uint32_t dst_smem,
                    const __nv_fp8_e4m3* __restrict__ hidden,
                    const int* sorted, int expert_id, int T_stride,
                    int Tk, int k_tile, int tid) {
  constexpr int CPR = K_TILE / 16;
  const int total = Tk * CPR;
  for (int c = tid; c < total; c += THREADS_PER_BLOCK) {
    int row = c / CPR, col = c % CPR;
    int token = sorted[expert_id * T_stride + row];
    uint32_t dst = dst_smem + row * K_TILE + col * 16;
    const auto* src = hidden + (size_t)token * H_DIM
                    + (size_t)k_tile * K_TILE + col * 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(dst), "l"(src));
  }
}

// Prefetch FP8 B tile (contiguous weight rows → SMEM)
__device__ __forceinline__
void prefetch_B_fp8(uint32_t dst_smem,
                    const __nv_fp8_e4m3* __restrict__ weight,
                    int stride, int k_tile, int tid) {
  constexpr int CPR = K_TILE / 16;
  constexpr int TOTAL = N_TILE * CPR;
  for (int c = tid; c < TOTAL; c += THREADS_PER_BLOCK) {
    int row = c / CPR, col = c % CPR;
    uint32_t dst = dst_smem + row * K_TILE + col * 16;
    const auto* src = weight + (size_t)row * stride
                    + (size_t)k_tile * K_TILE + col * 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(dst), "l"(src));
  }
}

// Prefetch BF16 A tile (intermediate buffer → SMEM)
__device__ __forceinline__
void prefetch_A_bf16(uint32_t dst_smem,
                     const nv_bfloat16* __restrict__ inter,
                     int expert_id, int T_stride,
                     int Tk, int k_tile, int tid) {
  constexpr int EPC = 8;   // 8 BF16 = 16 bytes per cp.async
  constexpr int CPR = K_TILE / EPC;
  const int total = Tk * CPR;
  for (int c = tid; c < total; c += THREADS_PER_BLOCK) {
    int row = c / CPR, col = c % CPR;
    uint32_t dst = dst_smem + (row * K_TILE + col * EPC) * sizeof(nv_bfloat16);
    const auto* src = inter + (size_t)expert_id * T_stride * I_DIM
                    + row * I_DIM + (size_t)k_tile * K_TILE + col * EPC;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(dst), "l"(src));
  }
}

// ============================================================================
// Kernel 1a: Routing — warp-cooperative top-k
//
// 1 warp = 1 token.  32 lanes split 256 experts (GROUP_SIZE == WARP_SIZE).
// Each iteration loads one group fully coalesced.  Top-2 per group via warp
// shuffle reduction, top-4 groups serial (only 8 values), global top-8 via
// 8 rounds of warp-level max reduction.
//
// Register budget: ~70 regs/thread (vs 512 in v1) → no spill.
// Grid: cdiv(T, NUM_WARPS) blocks × THREADS_PER_BLOCK threads.
// ============================================================================

static_assert(GROUP_SIZE == WARP_SIZE,
    "Warp-cooperative routing requires GROUP_SIZE == WARP_SIZE (32)");

__launch_bounds__(THREADS_PER_BLOCK)
__global__
void routing_topk_kernel(
    const float* __restrict__ routing_logits,   // [T, E_GLOBAL] FP32
    const float* __restrict__ routing_bias,     // [E_GLOBAL] FP32
    float  routed_scaling_factor,
    int    T,
    char*  scratch,
    size_t off_topk_idx,
    size_t off_topk_weights
) {
  int* g_topk_idx       = scratch_ptr<int>(scratch, off_topk_idx);
  float* g_topk_weights = scratch_ptr<float>(scratch, off_topk_weights);

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane    = threadIdx.x % WARP_SIZE;
  const int token   = blockIdx.x * NUM_WARPS + warp_id;

  if (token >= T) return;

  const float* logits_row = routing_logits + (size_t)token * E_GLOBAL;

  // ---- Step 1: Load groups, sigmoid+bias, group scores ----
  // Each iter loads 32 consecutive experts (one group) — fully coalesced.
  float my_sig[N_GROUP];
  float my_s_wb[N_GROUP];
  float group_scores[N_GROUP];

  #pragma unroll
  for (int g = 0; g < N_GROUP; g++) {
    int expert = g * GROUP_SIZE + lane;
    float logit = logits_row[expert];
    float bias  = routing_bias[expert];

    float sig  = 1.0f / (1.0f + fast_exp(-logit));
    float s_wb = sig + bias;
    my_sig[g]  = sig;
    my_s_wb[g] = s_wb;

    // Warp max reduction → top-1
    float top1 = s_wb;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      top1 = fmaxf(top1, __shfl_xor_sync(0xFFFFFFFF, top1, off));

    // Mask out top-1 lane, reduce again → top-2
    unsigned t1_mask  = __ballot_sync(0xFFFFFFFF, s_wb == top1);
    int      t1_lane  = __ffs(t1_mask) - 1;
    float    masked   = (lane == t1_lane) ? -FLT_MAX : s_wb;
    float    top2     = masked;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      top2 = fmaxf(top2, __shfl_xor_sync(0xFFFFFFFF, top2, off));

    group_scores[g] = top1 + top2;            // broadcast to all lanes
  }

  // ---- Step 2: Top-4 groups (serial, 8 values, warp-uniform) ----
  int   kept_groups[TOPK_GROUP];
  float kept_gscores[TOPK_GROUP];

  #pragma unroll
  for (int i = 0; i < TOPK_GROUP; i++) {
    kept_gscores[i] = -FLT_MAX;
    kept_groups[i]  = 0;
  }
  #pragma unroll
  for (int g = 0; g < N_GROUP; g++) {
    if (group_scores[g] > kept_gscores[TOPK_GROUP - 1]) {
      kept_gscores[TOPK_GROUP - 1] = group_scores[g];
      kept_groups[TOPK_GROUP - 1]  = g;
      #pragma unroll
      for (int i = TOPK_GROUP - 1; i > 0; i--) {
        if (kept_gscores[i] > kept_gscores[i - 1]) {
          float ts = kept_gscores[i]; kept_gscores[i] = kept_gscores[i-1]; kept_gscores[i-1] = ts;
          int   tg = kept_groups[i];  kept_groups[i]  = kept_groups[i-1];  kept_groups[i-1]  = tg;
        } else break;
      }
    }
  }

  // ---- Step 3: Extract 4 candidates per lane (register-only) ----
  // 4 kept groups × 32 lanes = 128 candidates total, 4 per lane, no SMEM.
  float cand_val[TOPK_GROUP];
  float cand_sig[TOPK_GROUP];
  int   cand_id[TOPK_GROUP];

  #pragma unroll
  for (int kg = 0; kg < TOPK_GROUP; kg++) {
    int g        = kept_groups[kg];
    cand_val[kg] = my_s_wb[g];
    cand_sig[kg] = my_sig[g];
    cand_id[kg]  = g * GROUP_SIZE + lane;
  }

  // ---- Step 4: Warp-level top-8 (8 rounds of max reduction) ----
  int   topk_experts[TOP_K];
  float topk_sig_out[TOP_K];

  #pragma unroll
  for (int k = 0; k < TOP_K; k++) {
    // Lane-local best among its 4 remaining candidates
    float best_val = -FLT_MAX;
    int   best_idx = 0;
    #pragma unroll
    for (int i = 0; i < TOPK_GROUP; i++) {
      if (cand_val[i] > best_val) {
        best_val = cand_val[i];
        best_idx = i;
      }
    }

    // Warp max reduction
    float global_max = best_val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      global_max = fmaxf(global_max, __shfl_xor_sync(0xFFFFFFFF, global_max, off));

    // Winner = lowest lane holding global_max
    unsigned winner_mask = __ballot_sync(0xFFFFFFFF, best_val == global_max);
    int      winner_lane = __ffs(winner_mask) - 1;

    // Broadcast winner's expert id & sigmoid (evaluated in each lane,
    // __shfl_sync returns the *winner lane's* value)
    int   sel_id  = cand_id[best_idx];
    float sel_sig = cand_sig[best_idx];
    topk_experts[k]  = __shfl_sync(0xFFFFFFFF, sel_id,  winner_lane);
    topk_sig_out[k]  = __shfl_sync(0xFFFFFFFF, sel_sig, winner_lane);

    // Invalidate the chosen candidate in the winner lane
    if (lane == winner_lane)
      cand_val[best_idx] = -FLT_MAX;
  }

  // ---- Step 5: Normalize weights & coalesced store ----
  float weight_sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < TOP_K; i++)
    weight_sum += topk_sig_out[i];
  float inv_sum = routed_scaling_factor / (weight_sum + 1e-20f);

  // Lanes 0..7 write one result each — coalesced within the warp
  if (lane < TOP_K) {
    g_topk_idx[token * TOP_K + lane]     = topk_experts[lane];
    g_topk_weights[token * TOP_K + lane] = topk_sig_out[lane] * inv_sum;
  }
}

// ============================================================================
// Kernel 1b: Routing — parallel permutation build
//
// All threads cooperatively build sorted_token_ids using atomics.
// Each thread handles one (token, k) pair in a grid-stride loop.
// ============================================================================

__launch_bounds__(THREADS_PER_BLOCK)
__global__
void routing_permute_kernel(
    int    T,
    int    local_expert_offset,
    char*  scratch,
    size_t off_topk_idx,
    size_t off_sorted_token_ids,
    size_t off_expert_token_count,
    size_t off_expert_token_slot
) {
  int* g_topk_idx   = scratch_ptr<int>(scratch, off_topk_idx);
  int* g_sorted     = scratch_ptr<int>(scratch, off_sorted_token_ids);
  int* g_count      = scratch_ptr<int>(scratch, off_expert_token_count);
  int* g_slot       = scratch_ptr<int>(scratch, off_expert_token_slot);

  int total = T * TOP_K;

  // Grid-stride over all token-expert assignments
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    int t = idx / TOP_K;
    int k = idx % TOP_K;
    int ge = g_topk_idx[t * TOP_K + k];
    int le = ge - local_expert_offset;
    if (le >= 0 && le < E_LOCAL) {
      int slot = atomicAdd(&g_slot[le], 1);
      g_sorted[le * T + slot] = t;
    }
  }

  // Copy final counts from slot counters
  __syncthreads();
  __threadfence();
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E_LOCAL;
       e += gridDim.x * blockDim.x) {
    g_count[e] = g_slot[e];
  }
}

// ============================================================================
// SMEM layouts (unchanged from v1)
// ============================================================================

// Phase 1 SMEM layout — tcgen05 path (B200-native)
// gate_acc REMOVED: accumulator now lives in TMEM (Tensor Memory), not SMEM.
// Two mbar slots for double-buffered TMA: mbar[0] guards buf[0], mbar[1] guards buf[1].
struct Phase1SMEM {
  alignas(128) __nv_fp8_e4m3 A_buf[2][M_TILE][K_TILE];   // 2 × 16 KB = 32 KB (FP8)
  alignas(128) __nv_fp8_e4m3 B_buf[2][N_TILE][K_TILE];   // 2 × 16 KB = 32 KB (FP8)
  alignas(8)   uint64_t       mbar[2];                     // TMA completion barriers
};  // Total: ~64 KB

// Phase 2 SMEM layout
struct Phase2SMEM {
  alignas(128) nv_bfloat16   A_buf[2][M_TILE][K_TILE];   // 2 × 32 KB = 64 KB (BF16)
  alignas(128) __nv_fp8_e4m3 B_buf[2][N_TILE][K_TILE];   // 2 × 16 KB = 32 KB (FP8)
  alignas(8)   uint64_t      mbar[2];
};  // Total: ~96 KB

union KernelSMEM {
  Phase1SMEM p1;
  Phase2SMEM p2;
};
// Union size = max(64, 96) = 96 KB → floor(228 KB / 96 KB) = 2 blocks/SM

// ============================================================================
// Kernel 2: Persistent GEMM — Phase 1 + Phase 2
//
// Regular launch (NOT cooperative). Work-stealing with atomic counters.
// Phase 2 uses claim/unclaim to skip unready experts instead of spinning.
// ============================================================================

// TMA descriptors passed as __grid_constant__: hardware copies to constant memory,
// shared by all threads in the grid without per-thread copies or register usage.
//
// Design: A-matrix (activations) uses raw pointer + per-thread cp.async scatter
//         because tokens are non-contiguous in global memory (indexed via sorted_token_ids).
//         B-matrix (weights) uses TMA: rows are contiguous [N_TILE × H_DIM] rectangles.
//         Phase-2 A (intermediate) uses TMA: stored contiguously per expert after Phase 1.
__launch_bounds__(THREADS_PER_BLOCK)
__global__
void persistent_gemm_kernel(
    // TMA descriptors for B-matrices and Phase-2 A (contiguous tensors)
    const __grid_constant__ CUtensorMap tma_w1,    // GEMM1 weights [E_LOCAL*2*I_DIM, H_DIM] FP8
    const __grid_constant__ CUtensorMap tma_w2,    // GEMM2 weights [E_LOCAL*H_DIM, I_DIM] FP8
    const __grid_constant__ CUtensorMap tma_inter, // intermediate [E_LOCAL*T, I_DIM] BF16
    // Raw pointer for scattered A-matrix loads (indexed by sorted_token_ids)
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float*         __restrict__ hidden_states_scale,
    const float*         __restrict__ gemm1_weights_scale,
    const float*         __restrict__ gemm2_weights_scale,
    int    T,
    int    local_expert_offset,
    char*  scratch,
    size_t off_topk_idx,
    size_t off_topk_weights,
    size_t off_sorted_token_ids,
    size_t off_expert_token_count,
    size_t off_gemm1_work_counter,
    size_t off_gemm2_work_counter,
    size_t off_gemm2_done_counter,
    size_t off_gemm1_done,
    size_t off_gemm2_claimed,
    size_t off_intermediate,
    size_t off_output
) {
  extern __shared__ char smem_raw[];
  const int tid = threadIdx.x;

  // Scratch pointers
  int*         g_sorted       = scratch_ptr<int>(scratch, off_sorted_token_ids);
  int*         g_count        = scratch_ptr<int>(scratch, off_expert_token_count);
  int*         g_p1_work_ctr  = scratch_ptr<int>(scratch, off_gemm1_work_counter);
  int*         g_p2_work_ctr  = scratch_ptr<int>(scratch, off_gemm2_work_counter);
  int*         g_p2_done_ctr  = scratch_ptr<int>(scratch, off_gemm2_done_counter);
  int*         g_done1        = scratch_ptr<int>(scratch, off_gemm1_done);
  int*         g_p2_claimed   = scratch_ptr<int>(scratch, off_gemm2_claimed);
  int*         g_topk_idx     = scratch_ptr<int>(scratch, off_topk_idx);
  float*       g_topk_wts     = scratch_ptr<float>(scratch, off_topk_weights);
  nv_bfloat16* g_intermediate = scratch_ptr<nv_bfloat16>(scratch, off_intermediate);
  float*       g_output       = scratch_ptr<float>(scratch, off_output);

  KernelSMEM& smem = *reinterpret_cast<KernelSMEM*>(smem_raw);

  // ==================================================================
  // Phase 1: drain all GEMM1+SwiGLU work
  // ==================================================================
  while (true) {
    __shared__ int s_work_id;
    if (tid == 0) s_work_id = atomicAdd(g_p1_work_ctr, 1);
    __syncthreads();
    if (s_work_id >= GEMM1_WORK_UNITS) break;

    int work_id   = s_work_id;
    int expert_id = work_id / GEMM1_N_PAIRS;
    int pair_j    = work_id % GEMM1_N_PAIRS;

    int Tk = g_count[expert_id];
    if (Tk == 0) {
      // Signal completion even for empty experts
      if (tid == 0) atomicAdd(&g_done1[expert_id], 1);
      __syncthreads();
      continue;
    }
    int Tk_clamped = min(Tk, M_TILE);

    // TMA row coordinates in weight tensor [E_LOCAL*2*I_DIM, H_DIM]:
    //   Gate rows: expert_id * 2*I_DIM + pair_j * N_TILE
    //   Up rows:   expert_id * 2*I_DIM + I_DIM + pair_j * N_TILE
    int gate_row_start  = expert_id * 2 * I_DIM + pair_j * N_TILE;
    int up_row_start    = expert_id * 2 * I_DIM + I_DIM + pair_j * N_TILE;
    int gate_scale_row  = pair_j;
    int up_scale_row    = GEMM1_N_PAIRS + pair_j;
    const float* w_scale_base = gemm1_weights_scale
        + (size_t)expert_id * NUM_2I_BLOCKS * NUM_H_BLOCKS;

    // Token row offset in the sorted activation layout.
    // prefetch_A_fp8 uses sorted_token_ids to scatter-load per-token rows.
    // TMA cannot scatter, so A is kept on the cp.async path.
    Phase1SMEM& p1 = smem.p1;

    // Byte counts for one A tile (cp.async, scattered) and one B tile (TMA)
    constexpr uint32_t B_TILE_BYTES = N_TILE * K_TILE * sizeof(__nv_fp8_e4m3); // 16 KB

    // =========================================================================
    // Gate sub-GEMM
    // ─────────────────────────────────────────────────────────────────────────
    // Synchronization split by path:
    //   A tiles: double-buffered cp.async → cp_async_wait_group (unchanged)
    //   B tiles: TMA + mbar per buffer slot → mbar_wait
    //
    // Two mbar slots: mbar[0] guards B_buf[0], mbar[1] guards B_buf[1].
    // A and B tiles share the same buf index for alignment.
    //
    // tcgen05.mma replaces mma.sync + manual ldmatrix fragment loads:
    //   Old: each warp loads A/B fragments into registers, issues mma.sync (1 warp)
    //   New: all 4 warps issue tcgen05.mma together (1 warpgroup), result in TMEM
    //
    // Warp division within the warpgroup (M=128):
    //   Warp 0 (lanes 0–31):   M-rows 0–31
    //   Warp 1 (lanes 32–63):  M-rows 32–63
    //   Warp 2 (lanes 64–95):  M-rows 64–95
    //   Warp 3 (lanes 96–127): M-rows 96–127
    // =========================================================================

    // Allocate TMEM for gate accumulator [M=128, N=128, FP32]
    // All 128 threads call this simultaneously (sync.aligned).
    uint32_t tmem_gate = tcgen05_alloc<TMEM_COLS>();

    // Init mbar for both buffer slots (thread 0 only)
    if (tid == 0) { mbar_init(&p1.mbar[0]); mbar_init(&p1.mbar[1]); }
    __syncthreads();
    uint32_t parity_g[2] = {0, 0};

    // ── Prologue: prefetch tile 0 ─────────────────────────────────────────
    // A: cp.async scatter (hidden_states → A_buf[0])
    prefetch_A_fp8(smem_addr_of(&p1.A_buf[0][0][0]),
                   hidden_states, g_sorted, expert_id, T, Tk_clamped, 0, tid);
    cp_async_commit();
    // B: TMA → B_buf[0], mbar[0] signals completion
    if (tid == 0) {
      mbar_arrive_expect_tx(&p1.mbar[0], B_TILE_BYTES);
      tma_load_2d(smem_addr_of(&p1.B_buf[0][0][0]),
                  &tma_w1, /*col=*/0, /*row=*/gate_row_start, &p1.mbar[0]);
    }

    for (int kt = 0; kt < GEMM1_K_TILES; kt++) {
      int buf  = kt & 1;
      int nbuf = 1 - buf;

      // ── Prefetch next tile ──────────────────────────────────────────────
      if (kt + 1 < GEMM1_K_TILES) {
        // A: cp.async into A_buf[nbuf]
        prefetch_A_fp8(smem_addr_of(&p1.A_buf[nbuf][0][0]),
                       hidden_states, g_sorted, expert_id, T, Tk_clamped, kt+1, tid);
        cp_async_commit();
        // B: TMA into B_buf[nbuf], mbar[nbuf] signals when B ready
        if (tid == 0) {
          mbar_init(&p1.mbar[nbuf]);  // reset for reuse
          mbar_arrive_expect_tx(&p1.mbar[nbuf], B_TILE_BYTES);
          tma_load_2d(smem_addr_of(&p1.B_buf[nbuf][0][0]),
                      &tma_w1, (kt+1) * K_TILE, gate_row_start, &p1.mbar[nbuf]);
        }
      }

      // ── Wait for current tile ───────────────────────────────────────────
      // A: cp.async group (current tile either just committed or 1 behind)
      if (kt + 1 < GEMM1_K_TILES)
        cp_async_wait_group<1>();   // current A ready; next A can still be in flight
      else
        cp_async_wait_group<0>();
      // B: mbar waits until TMA has written all B_TILE_BYTES to B_buf[buf]
      mbar_wait(&p1.mbar[buf], parity_g[buf]);
      parity_g[buf] ^= 1;
      // tcgen05.mma is a CTA-collective: ALL 128 threads must issue it together.
      // mbar_wait exits per-thread at slightly different times → barrier needed.
      __syncthreads();

      // ── tcgen05 MMA ────────────────────────────────────────────────────
      // Build SMEM descriptors for this buf's A and B tiles.
      // 128B swizzle in descriptor must match TMA's CU_TENSOR_MAP_SWIZZLE_128B.
      uint64_t a_desc = make_smem_desc(&p1.A_buf[buf][0][0],
                                        K_TILE * sizeof(__nv_fp8_e4m3));
      uint64_t b_desc = make_smem_desc(&p1.B_buf[buf][0][0],
                                        K_TILE * sizeof(__nv_fp8_e4m3));

      // Two MMA steps of K=64 cover K_TILE=128:
      //   Step 0 (K offset 0..63):   accumulate=false → initialize TMEM
      //   Step 1 (K offset 64..127): accumulate=true  → add to TMEM
      for (int ks = 0; ks < K_TILE / TG05_K_STEP; ks++) {
        // Advance K-offset in descriptor: bits [13:0] += k_byte_off >> 4
        uint32_t k_byte_off = ks * TG05_K_STEP;  // offset in bytes for FP8
        uint64_t ak = a_desc | (static_cast<uint64_t>(k_byte_off >> 4) & 0x3FFF);
        uint64_t bk = b_desc | (static_cast<uint64_t>(k_byte_off >> 4) & 0x3FFF);
        tcgen05_mma_fp8(tmem_gate, ak, bk, (kt > 0 || ks > 0));
      }

      // Apply FP8 weight scale into TMEM after this K-tile lands
      float w_scale = w_scale_base[gate_scale_row * NUM_H_BLOCKS + kt];
      if (w_scale != 1.0f) {
        tcgen05_commit(tmem_gate);
        tcgen05_fence();
        // Each thread t owns M-row t. Scale all N=128 output values for this row.
        for (int nc = 0; nc < N_TILE; nc += 4) {
          uint32_t col_off = nc * (THREADS_PER_BLOCK * sizeof(float));
          uint32_t v[4];
          tcgen05_ld_4(v, tmem_gate, col_off);
          float* f = reinterpret_cast<float*>(v);
          f[0] *= w_scale; f[1] *= w_scale; f[2] *= w_scale; f[3] *= w_scale;
          asm volatile(
            "tcgen05.st.sync.aligned.4x32b [%0], {%1, %2, %3, %4};"
            :: "r"(tmem_gate + col_off), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
            : "memory");
        }
      }
    }
    // Final commit+fence: tmem_gate is stable
    tcgen05_commit(tmem_gate);
    tcgen05_fence();

    // =========================================================================
    // Up sub-GEMM — same structure, B from tma_w1 at up_row_start
    // =========================================================================

    uint32_t tmem_up = tcgen05_alloc<TMEM_COLS>();

    if (tid == 0) { mbar_init(&p1.mbar[0]); mbar_init(&p1.mbar[1]); }
    __syncthreads();
    uint32_t parity_u[2] = {0, 0};

    prefetch_A_fp8(smem_addr_of(&p1.A_buf[0][0][0]),
                   hidden_states, g_sorted, expert_id, T, Tk_clamped, 0, tid);
    cp_async_commit();
    if (tid == 0) {
      mbar_arrive_expect_tx(&p1.mbar[0], B_TILE_BYTES);
      tma_load_2d(smem_addr_of(&p1.B_buf[0][0][0]),
                  &tma_w1, 0, up_row_start, &p1.mbar[0]);
    }

    for (int kt = 0; kt < GEMM1_K_TILES; kt++) {
      int buf  = kt & 1;
      int nbuf = 1 - buf;

      if (kt + 1 < GEMM1_K_TILES) {
        prefetch_A_fp8(smem_addr_of(&p1.A_buf[nbuf][0][0]),
                       hidden_states, g_sorted, expert_id, T, Tk_clamped, kt+1, tid);
        cp_async_commit();
        if (tid == 0) {
          mbar_init(&p1.mbar[nbuf]);
          mbar_arrive_expect_tx(&p1.mbar[nbuf], B_TILE_BYTES);
          tma_load_2d(smem_addr_of(&p1.B_buf[nbuf][0][0]),
                      &tma_w1, (kt+1) * K_TILE, up_row_start, &p1.mbar[nbuf]);
        }
      }

      if (kt + 1 < GEMM1_K_TILES) cp_async_wait_group<1>();
      else                         cp_async_wait_group<0>();
      mbar_wait(&p1.mbar[buf], parity_u[buf]);
      parity_u[buf] ^= 1;
      __syncthreads();

      uint64_t a_desc = make_smem_desc(&p1.A_buf[buf][0][0],
                                        K_TILE * sizeof(__nv_fp8_e4m3));
      uint64_t b_desc = make_smem_desc(&p1.B_buf[buf][0][0],
                                        K_TILE * sizeof(__nv_fp8_e4m3));

      for (int ks = 0; ks < K_TILE / TG05_K_STEP; ks++) {
        uint32_t k_byte_off = ks * TG05_K_STEP;
        uint64_t ak = a_desc | (static_cast<uint64_t>(k_byte_off >> 4) & 0x3FFF);
        uint64_t bk = b_desc | (static_cast<uint64_t>(k_byte_off >> 4) & 0x3FFF);
        tcgen05_mma_fp8(tmem_up, ak, bk, (kt > 0 || ks > 0));
      }

      float w_scale = w_scale_base[up_scale_row * NUM_H_BLOCKS + kt];
      if (w_scale != 1.0f) {
        tcgen05_commit(tmem_up);
        tcgen05_fence();
        for (int nc = 0; nc < N_TILE; nc += 4) {
          uint32_t col_off = nc * (THREADS_PER_BLOCK * sizeof(float));
          uint32_t v[4];
          tcgen05_ld_4(v, tmem_up, col_off);
          float* f = reinterpret_cast<float*>(v);
          f[0] *= w_scale; f[1] *= w_scale; f[2] *= w_scale; f[3] *= w_scale;
          asm volatile(
            "tcgen05.st.sync.aligned.4x32b [%0], {%1, %2, %3, %4};"
            :: "r"(tmem_up + col_off), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
            : "memory");
        }
      }
    }
    tcgen05_commit(tmem_up);
    tcgen05_fence();

    // ---- Fused SwiGLU — reads gate and up from TMEM ----
    //
    // Thread tid == M-row tid (0..127). Reads N=128 gate and up values from TMEM,
    // computes SwiGLU(up) * gate, stores BF16 result to g_intermediate.
    //
    // tcgen05_ld_4 reads 4 consecutive N-values at a time from this thread's TMEM row.
    // TMEM layout: column c → byte offset c * (THREADS_PER_BLOCK * 4).
    //
    // Only threads with tid < Tk_clamped have valid token data; others still run
    // the reads (TMEM is allocated for M=128) but skip the global store.
    {
      bool row_valid = (tid < Tk_clamped);
      size_t inter_row = row_valid ?
          (size_t)expert_id * T * I_DIM + tid * I_DIM + pair_j * N_TILE : 0;

      for (int nc = 0; nc < N_TILE; nc += 4) {
        uint32_t col_off = nc * (THREADS_PER_BLOCK * sizeof(float));
        // Read gate[m=tid][nc..nc+3] and up[m=tid][nc..nc+3] from TMEM
        uint32_t gv[4], uv[4];
        tcgen05_ld_4(gv, tmem_gate, col_off);
        tcgen05_ld_4(uv, tmem_up,   col_off);

        if (row_valid) {
          // SwiGLU: out[n] = silu(up[n]) * gate[n]
          //         silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
          float* g = reinterpret_cast<float*>(gv);
          float* u = reinterpret_cast<float*>(uv);
          for (int i = 0; i < 4; i++) {
            float silu_u = u[i] / (1.0f + fast_exp(-u[i]));
            float out    = silu_u * g[i];
            g_intermediate[inter_row + nc + i] = __float2bfloat16(out);
          }
        }
      }
    }

    // Release TMEM allocations for next work unit
    tcgen05_dealloc<TMEM_COLS>(tmem_gate);
    tcgen05_dealloc<TMEM_COLS>(tmem_up);

    // Signal completion
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      atomicAdd(&g_done1[expert_id], 1);
    }
  }

  // ==================================================================
  // Phase 2: GEMM2 + Weighted Accumulate (work-skip, no spin-wait)
  // ==================================================================
  Phase2SMEM& p2 = smem.p2;

  while (true) {
    // Check global termination (volatile load — no RMW traffic)
    __shared__ int s_all_done;
    if (tid == 0) s_all_done = *((volatile int*)g_p2_done_ctr);
    __syncthreads();
    if (s_all_done >= GEMM2_WORK_UNITS) break;

    // Grab next work index (wraps around via modular arithmetic)
    __shared__ int s_raw_id;
    if (tid == 0) s_raw_id = atomicAdd(g_p2_work_ctr, 1);
    __syncthreads();
    int work_id = s_raw_id % GEMM2_WORK_UNITS;

    // Try to claim this slot (exactly-once execution)
    __shared__ int s_claimed;
    if (tid == 0) s_claimed = atomicCAS(&g_p2_claimed[work_id], 0, 1);
    __syncthreads();
    if (s_claimed != 0) continue;  // already taken

    int expert_id = work_id / GEMM2_N_TILES;
    int n_tile_j  = work_id % GEMM2_N_TILES;
    int Tk = g_count[expert_id];

    // Empty expert — mark done and skip
    if (Tk == 0) {
      if (tid == 0) atomicAdd(g_p2_done_ctr, 1);
      __syncthreads();
      continue;
    }

    int Tk_clamped = min(Tk, M_TILE);

    // Check if Phase 1 is complete for this expert (volatile load, no RMW)
    __shared__ int s_ready;
    if (tid == 0) s_ready = (*((volatile int*)&g_done1[expert_id]) >= GEMM1_N_PAIRS);
    __syncthreads();
    if (!s_ready) {
      // Unclaim so another pass can retry
      if (tid == 0) atomicExch(&g_p2_claimed[work_id], 0);
      __syncthreads();
      __nanosleep(200);  // longer yield — lets Phase 1 make progress
      continue;
    }

    // ---- GEMM2: TMA (B) + tcgen05 MMA → TMEM → weighted accumulate ----
    // A: BF16 intermediate [E_LOCAL*T, I_DIM] — contiguous per expert, TMA loads it
    // B: FP8 GEMM2 weights [E_LOCAL*H_DIM, I_DIM] — TMA loads [N_TILE × K_TILE] box
    const float* w2_scale = gemm2_weights_scale
        + (size_t)expert_id * NUM_H_BLOCKS * NUM_I_BLOCKS;

    // TMA row coords: intermediate A at expert_id*T, B weight at expert_id*H_DIM + n_tile_j*N_TILE
    int inter_row_base = expert_id * T;
    int w2_row_start   = expert_id * H_DIM + n_tile_j * N_TILE;

    constexpr uint32_t A2_TILE_BYTES = M_TILE * K_TILE * sizeof(nv_bfloat16);
    constexpr uint32_t B2_TILE_BYTES = N_TILE * K_TILE * sizeof(__nv_fp8_e4m3);

    uint32_t tmem_out = tcgen05_alloc<TMEM_COLS>();

    if (tid == 0) { mbar_init(&p2.mbar[0]); mbar_init(&p2.mbar[1]); }
    __syncthreads();
    uint32_t parity_p2[2] = {0, 0};

    // Prologue: prefetch tile 0 for both A and B via TMA
    if (tid == 0) {
      mbar_arrive_expect_tx(&p2.mbar[0], A2_TILE_BYTES + B2_TILE_BYTES);
      tma_load_2d(smem_addr_of(&p2.A_buf[0][0][0]),
                  &tma_inter, 0, inter_row_base, &p2.mbar[0]);
      tma_load_2d(smem_addr_of(&p2.B_buf[0][0][0]),
                  &tma_w2, 0, w2_row_start, &p2.mbar[0]);
    }

    for (int kt = 0; kt < GEMM2_K_TILES; kt++) {
      int buf  = kt & 1;
      int nbuf = 1 - buf;

      if (kt + 1 < GEMM2_K_TILES) {
        if (tid == 0) {
          mbar_init(&p2.mbar[nbuf]);
          mbar_arrive_expect_tx(&p2.mbar[nbuf], A2_TILE_BYTES + B2_TILE_BYTES);
          tma_load_2d(smem_addr_of(&p2.A_buf[nbuf][0][0]),
                      &tma_inter, (kt+1) * K_TILE, inter_row_base, &p2.mbar[nbuf]);
          tma_load_2d(smem_addr_of(&p2.B_buf[nbuf][0][0]),
                      &tma_w2, (kt+1) * K_TILE, w2_row_start, &p2.mbar[nbuf]);
        }
      }

      // Both A and B use TMA — wait on the single mbar for this buf
      mbar_wait(&p2.mbar[buf], parity_p2[buf]);
      parity_p2[buf] ^= 1;
      __syncthreads();  // tcgen05.mma is CTA-collective; mbar_wait exits per-thread

      // tcgen05 MMA: BF16 A × FP8 B → FP32 in TMEM
      // NOTE: tcgen05 kind::f8f6f4 expects FP8 operands; for BF16 A, use kind::bf16
      // Here we treat A as BF16 input — the MMA variant for BF16 × FP8 mixed precision
      uint64_t a_desc = make_smem_desc(&p2.A_buf[buf][0][0],
                                        K_TILE * sizeof(nv_bfloat16));
      uint64_t b_desc = make_smem_desc(&p2.B_buf[buf][0][0],
                                        K_TILE * sizeof(__nv_fp8_e4m3));

      for (int ks = 0; ks < K_TILE / TG05_K_STEP; ks++) {
        uint32_t ka = ks * TG05_K_STEP * sizeof(nv_bfloat16);
        uint32_t kb = ks * TG05_K_STEP * sizeof(__nv_fp8_e4m3);
        uint64_t ak = a_desc | (static_cast<uint64_t>(ka >> 4) & 0x3FFF);
        uint64_t bk = b_desc | (static_cast<uint64_t>(kb >> 4) & 0x3FFF);
        // scale_d must be literal 0 or 1 in PTX — not a register
        if (kt > 0 || ks > 0) {
          asm volatile(
            "tcgen05.mma.cta_group::1.kind::bf16 [%0], %1, %2, 1;"
            :: "r"(tmem_out), "l"(ak), "l"(bk) : "memory");
        } else {
          asm volatile(
            "tcgen05.mma.cta_group::1.kind::bf16 [%0], %1, %2, 0;"
            :: "r"(tmem_out), "l"(ak), "l"(bk) : "memory");
        }
      }

      float w_scale = w2_scale[n_tile_j * NUM_I_BLOCKS + kt];
      if (w_scale != 1.0f) {
        tcgen05_commit(tmem_out);
        tcgen05_fence();
        for (int nc = 0; nc < N_TILE; nc += 4) {
          uint32_t col_off = nc * (THREADS_PER_BLOCK * sizeof(float));
          uint32_t v[4];
          tcgen05_ld_4(v, tmem_out, col_off);
          float* f = reinterpret_cast<float*>(v);
          f[0] *= w_scale; f[1] *= w_scale; f[2] *= w_scale; f[3] *= w_scale;
          asm volatile(
            "tcgen05.st.sync.aligned.4x32b [%0], {%1, %2, %3, %4};"
            :: "r"(tmem_out + col_off), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
            : "memory");
        }
      }
    }
    tcgen05_commit(tmem_out);
    tcgen05_fence();

    // ---- Weighted accumulate from TMEM → global output ----
    // Thread tid owns M-row tid. Read N=128 output values, apply routing weight,
    // atomicAdd into g_output[token_id][col].
    {
      bool row_valid = (tid < Tk_clamped);
      int token_id   = row_valid ? g_sorted[expert_id * T + tid] : -1;
      int ge         = local_expert_offset + expert_id;
      float route_w  = 0.0f;
      if (row_valid) {
        for (int k = 0; k < TOP_K; k++) {
          if (g_topk_idx[token_id * TOP_K + k] == ge) {
            route_w = g_topk_wts[token_id * TOP_K + k];
            break;
          }
        }
      }

      int col_base = n_tile_j * N_TILE;
      for (int nc = 0; nc < N_TILE; nc += 4) {
        uint32_t col_off = nc * (THREADS_PER_BLOCK * sizeof(float));
        uint32_t v[4];
        tcgen05_ld_4(v, tmem_out, col_off);
        if (row_valid && route_w != 0.0f) {
          float* f = reinterpret_cast<float*>(v);
          atomicAdd(&g_output[token_id * H_DIM + col_base + nc + 0], f[0] * route_w);
          atomicAdd(&g_output[token_id * H_DIM + col_base + nc + 1], f[1] * route_w);
          atomicAdd(&g_output[token_id * H_DIM + col_base + nc + 2], f[2] * route_w);
          atomicAdd(&g_output[token_id * H_DIM + col_base + nc + 3], f[3] * route_w);
        }
      }
    }

    tcgen05_dealloc<TMEM_COLS>(tmem_out);

    // Mark this Phase 2 unit as completed
    if (tid == 0) atomicAdd(g_p2_done_ctr, 1);
    __syncthreads();
  }
}

// ============================================================================
// Host launcher v2
// ============================================================================

void launch_persistent_moe_v2(
    const float*          routing_logits,
    const float*          routing_bias,
    const __nv_fp8_e4m3*  hidden_states,
    const float*          hidden_states_scale,
    const __nv_fp8_e4m3*  gemm1_weights,
    const float*          gemm1_weights_scale,
    const __nv_fp8_e4m3*  gemm2_weights,
    const float*          gemm2_weights_scale,
    int    T,
    int    local_expert_offset,
    float  routed_scaling_factor,
    float* output,          // [T, H_DIM] FP32 — caller-allocated
    cudaStream_t stream
) {
  ScratchOffsets2 off = compute_scratch_offsets2(T);

  // Allocate and zero scratch
  char* scratch = nullptr;
  CUDA_CHECK(cudaMallocAsync(&scratch, off.total_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(scratch, 0, off.total_bytes, stream));

  // --- Kernel 1a: Routing top-k (1 warp = 1 token) ---
  int routing_blocks = cdiv(T, NUM_WARPS);
  routing_topk_kernel<<<routing_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      routing_logits, routing_bias, routed_scaling_factor,
      T, scratch,
      off.topk_idx, off.topk_weights
  );
  CUDA_CHECK(cudaGetLastError());

  // --- Kernel 1b: Permutation build ---
  int permute_blocks = cdiv(T * TOP_K, THREADS_PER_BLOCK);
  routing_permute_kernel<<<permute_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      T, local_expert_offset, scratch,
      off.topk_idx, off.sorted_token_ids,
      off.expert_token_count, off.expert_token_slot
  );
  CUDA_CHECK(cudaGetLastError());

  // --- Kernel 2: Persistent GEMM (TMA + tcgen05) ---

  // Build TMA descriptors on the host. Each descriptor is 128 bytes and
  // encodes the full tensor layout + swizzle mode for the hardware DMA engine.
  //
  // tma_w1: GEMM1 weights [E_LOCAL * 2*I_DIM, H_DIM] — gate rows then up rows
  // tma_w2: GEMM2 weights [E_LOCAL * H_DIM, I_DIM]
  // tma_inter: intermediate BF16 [E_LOCAL * T, I_DIM] (written by Phase 1)
  CUtensorMap tma_w1_desc, tma_w2_desc, tma_inter_desc;

  // Intermediate buffer (Phase 2 A) — points into scratch allocation.
  // We treat it as [E_LOCAL * T, I_DIM] BF16.
  nv_bfloat16* inter_ptr = scratch_ptr<nv_bfloat16>(scratch, off.intermediate);

  create_tma_fp8 (&tma_w1_desc,
      gemm1_weights,
      (uint64_t)E_LOCAL * 2 * I_DIM,   // rows
      (uint64_t)H_DIM,                  // cols
      (uint32_t)N_TILE,                 // tile rows
      (uint32_t)K_TILE);                // tile cols

  create_tma_fp8 (&tma_w2_desc,
      gemm2_weights,
      (uint64_t)E_LOCAL * H_DIM,
      (uint64_t)I_DIM,
      (uint32_t)N_TILE,
      (uint32_t)K_TILE);

  create_tma_bf16(&tma_inter_desc,
      inter_ptr,
      (uint64_t)E_LOCAL * T,
      (uint64_t)I_DIM,
      (uint32_t)M_TILE,
      (uint32_t)K_TILE);

  // Set max SMEM per block (96 KB > 48 KB default)
  int smem_bytes = (int)sizeof(KernelSMEM);
  CUDA_CHECK(cudaFuncSetAttribute(
      persistent_gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes));

  // Query occupancy for SMEM-limited launch: floor(228 KB / 96 KB) = 2 blocks/SM
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm, persistent_gemm_kernel,
      THREADS_PER_BLOCK, smem_bytes));

  int gemm_blocks = NUM_SMS * max(max_blocks_per_sm, 1);
  gemm_blocks = min(gemm_blocks, GEMM1_WORK_UNITS + GEMM2_WORK_UNITS);

  persistent_gemm_kernel<<<gemm_blocks, THREADS_PER_BLOCK, smem_bytes, stream>>>(
      tma_w1_desc, tma_w2_desc, tma_inter_desc,
      hidden_states, hidden_states_scale,
      gemm1_weights_scale, gemm2_weights_scale,
      T, local_expert_offset,
      scratch,
      off.topk_idx, off.topk_weights,
      off.sorted_token_ids, off.expert_token_count,
      off.gemm1_work_counter, off.gemm2_work_counter,
      off.gemm2_done_counter, off.gemm1_done,
      off.gemm2_claimed,
      off.intermediate, off.output
  );
  CUDA_CHECK(cudaGetLastError());

  // Copy output
  float* scratch_output = reinterpret_cast<float*>(scratch + off.output);
  CUDA_CHECK(cudaMemcpyAsync(
      output, scratch_output,
      (size_t)T * H_DIM * sizeof(float),
      cudaMemcpyDeviceToDevice, stream));

  CUDA_CHECK(cudaFreeAsync(scratch, stream));
}
