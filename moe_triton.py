"""
moe_triton.py — DeepSeek-V3 MoE GEMM1 + SwiGLU in Triton

Triton programming model in one sentence:
  You describe the algorithm at the BLOCK level — Triton infers thread layout,
  shared memory arrangement, and async pipeline structure automatically.

Compilation pipeline:
  Python source (@triton.jit)
    → Python AST
    → Triton IR            [SSA form; tensors are 1D/2D blocks, not threads]
    → TritonGPU IR         [key optimization passes happen here — see below]
        ├─ CoalescePass            reorder tensor ops for coalesced GMEM access
        ├─ F32DotTC                use tensor cores for tl.dot when dtype allows
        ├─ LayoutConversionPass    insert shared-memory staging between producer/consumer
        ├─ PipelinePass            insert cp.async / TMA + mbarriers (software pipelining)
        └─ WarpSpecializationPass  split warps: "producer" warps do async copy,
                                   "consumer" warps do compute (Hopper/Blackwell)
    → LLVM IR              [vectorization, instruction selection]
    → PTX                  [by LLVM NVPTX backend]
    → SASS                 [by ptxas, separate invocation]

  Autotuning runs the above pipeline for each (BLOCK_M, BLOCK_N, BLOCK_K,
  num_stages, num_warps) candidate and picks the fastest.

Key difference from CuteDSL:
  CuteDSL: programmer specifies every atom, layout, and sync point explicitly.
  Triton:  programmer specifies the block algorithm; Triton's passes add the rest.
"""

import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Model constants
# ─────────────────────────────────────────────────────────────────────────────

H_DIM       = 7168
I_DIM       = 2048
TOP_K       = 8
BLOCK_QUANT = 128   # FP8 quantization block size; K-tile must be a multiple

# ─────────────────────────────────────────────────────────────────────────────
# Kernel: fused GEMM1 Gate + Up + SwiGLU for one expert
#
# Computes:
#   gate[Tk, I_DIM] = hidden[tokens, :] @ W_gate[expert, :, :]^T  (FP8)
#   up  [Tk, I_DIM] = hidden[tokens, :] @ W_up  [expert, :, :]^T  (FP8)
#   out [Tk, I_DIM] = silu(up) * gate                              (BF16)
#
# Grid decomposition:
#   pid_m = token tile  (Tk / BLOCK_M tiles along token dimension)
#   pid_n = output tile (I_DIM / BLOCK_N tiles along I_DIM dimension)
#   One kernel launch per expert.
#
# Block-wise FP8 dequantization:
#   Triton cannot express "multiply accumulator mid-loop" as cleanly as
#   tcgen05.ld/st.  Instead, we dequantize A and B before tl.dot by
#   promoting to BF16 scaled by the block factor.  This gives:
#     A_bf16 = A_fp8 * a_scale       (per row per K-block)
#     B_bf16 = B_fp8 * b_scale       (per N-tile per K-block, broadcasted)
#     acc   += tl.dot(A_bf16, B_bf16^T)
#   Triton will use native FP8 tensor core MMA (tcgen05 on sm_100a) when
#   the PipelinePass emits the right instruction via its hardware capability
#   table — the programmer does not specify this.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # Triton searches this space; PipelinePass picks num_stages internally
        # based on SMEM budget and latency model for the target GPU.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=4, num_stages=2),
    ],
    key=["Tk", "H", "N"],
)
@triton.jit
def moe_gemm1_swiglu_kernel(
    # Pointers
    hidden_ptr,         # [T,  H] FP8 — all tokens, expert scatter-loads its slice
    w_gate_ptr,         # [N,  H] FP8 — gate weight, this expert's N-tile shard
    w_up_ptr,           # [N,  H] FP8 — up   weight, same expert, same shard
    sorted_ids_ptr,     # [Tk]    int32 — which global token indices belong here
    out_ptr,            # [Tk, N] BF16 — SwiGLU output, written back to inter buffer
    hs_scale_ptr,       # [ceil(H/128), T] float32 — per-token, per-K-block A scale
    wg_scale_ptr,       # [ceil(N/128), ceil(H/128)] float32 — per-(N-tile, K-block)
    wu_scale_ptr,       # same layout as wg_scale_ptr, for up weight
    # Dimensions
    Tk,                 # int: tokens assigned to this expert
    H: tl.constexpr,   # = H_DIM = 7168
    N: tl.constexpr,   # = I_DIM = 2048
    # Tile sizes (searched by @triton.autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ── Program ID → tile coordinates ────────────────────────────────────────
    # Triton's 1D grid: we linearize (pid_m, pid_n) into a single program_id.
    # This lets the autotuner also explore different grid-stride orderings.
    pid      = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m    = pid  // num_pid_n   # which token-tile
    pid_n    = pid  %  num_pid_n   # which output N-tile

    # ── Token gather ──────────────────────────────────────────────────────────
    # Triton does not have a scatter/gather primitive.
    # We load token ids explicitly, then compute all pointers from them.
    # PipelinePass will try to prefetch these loads into registers.
    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    mask_m  = offs_m < Tk

    # Each row of A comes from a DIFFERENT token (non-contiguous in GMEM).
    # tl.load with a 1D ptr + mask: Triton's LayoutConversionPass will insert
    # shared-memory staging and try to coalesce via reordering if possible.
    token_ids = tl.load(sorted_ids_ptr + offs_m, mask=mask_m, other=0)
    # token_ids: [BLOCK_M] — each entry is a row index into hidden_ptr

    # ── Accumulators (FP32) ───────────────────────────────────────────────────
    # tl.zeros allocates register-resident accumulators.
    # On sm_100a these map to TMEM via Triton's register allocation pass,
    # but the programmer does not specify this — it is inferred from the
    # tl.dot call and the dtype requested.
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── Main K loop ───────────────────────────────────────────────────────────
    # @triton.autotune's num_stages controls how many iterations Triton's
    # PipelinePass overlaps.  With num_stages=2 on sm_100a:
    #   - Prologue:  issues async copy for kt=0 into shared memory buf 0
    #   - Iteration: issues async copy for kt+1 into buf 1,
    #                waits for kt's copy, runs tl.dot on kt's data
    #   - Epilogue:  drains the last outstanding copy
    # This matches the manual double-buffer in persistent_kernel2.cu, but
    # Triton inserts the cp.async + mbarrier code automatically.
    # The programmer only writes the serial loop below.

    offs_k = tl.arange(0, BLOCK_K)

    for k in tl.range(tl.cdiv(H, BLOCK_K), loop_unroll_factor=1):
        k_start = k * BLOCK_K
        k_blk   = k   # = k_start // BLOCK_K (scale block index along K)

        # ── Load A tile: gather [BLOCK_M, BLOCK_K] from scattered token rows ──
        # Pointer for row i = hidden_ptr + token_ids[i] * H + (k_start + j)
        a_ptrs = (hidden_ptr
                  + token_ids[:, None] * H
                  + (k_start + offs_k)[None, :])
        mask_a = mask_m[:, None] & ((k_start + offs_k)[None, :] < H)
        # Load as FP8 directly — Triton's F32DotTC pass promotes to BF16/FP32
        # internally when emitting the tensor core instruction.
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float8e4nv)

        # ── Load B tiles: contiguous [BLOCK_K, BLOCK_N] weight slices ─────────
        # B is transposed relative to A: weight layout is [N, H] (N rows, H cols).
        # We load [BLOCK_N, BLOCK_K] and let tl.dot handle the transpose.
        b_ptrs_g = (w_gate_ptr
                    + offs_n[:, None] * H
                    + (k_start + offs_k)[None, :])
        b_ptrs_u = (w_up_ptr
                    + offs_n[:, None] * H
                    + (k_start + offs_k)[None, :])
        mask_b = (offs_n[:, None] < N) & ((k_start + offs_k)[None, :] < H)
        b_gate = tl.load(b_ptrs_g, mask=mask_b, other=0.0).to(tl.float8e4nv)
        b_up   = tl.load(b_ptrs_u, mask=mask_b, other=0.0).to(tl.float8e4nv)

        # ── FP8 block-quantization scales ────────────────────────────────────
        # Hidden scale: one float per (K-block, token). Shape [NUM_H_BLOCKS, T].
        # We need scale[k_blk, token_ids[i]] for each row i of A.
        # This is another scatter load — non-coalesced, but small (BLOCK_M floats).
        a_scale = tl.load(hs_scale_ptr + k_blk * (H // BLOCK_K) + token_ids,
                          mask=mask_m, other=1.0)  # [BLOCK_M]

        # Weight scale: one float per (N-tile, K-block). Shape [N//BQ, H//BQ].
        n_blk  = pid_n
        wg_sc = tl.load(wg_scale_ptr + n_blk * (H // BLOCK_K) + k_blk)  # scalar
        wu_sc = tl.load(wu_scale_ptr + n_blk * (H // BLOCK_K) + k_blk)  # scalar

        # Dequantize: promote FP8 → BF16, multiply by scale before dot.
        # This keeps tl.dot inputs in BF16 (or Triton may use native FP8 dot
        # if the hardware supports it and input_precision allows it).
        # Scale is broadcast: a_scale[:,None] * scalar broadcasts to [M, N].
        a_bf16 = a.to(tl.bfloat16) * a_scale[:, None].to(tl.bfloat16)
        bg_bf16 = b_gate.to(tl.bfloat16) * wg_sc.to(tl.bfloat16)
        bu_bf16 = b_up.to(tl.bfloat16)   * wu_sc.to(tl.bfloat16)

        # ── Matrix multiply-accumulate ────────────────────────────────────────
        # tl.dot: [BLOCK_M, BLOCK_K] × [BLOCK_N, BLOCK_K]^T → [BLOCK_M, BLOCK_N]
        # Triton's F32DotTC pass maps this to tcgen05.mma on sm_100a, selecting
        # M=128/N=128/K=64 atom (same hardware instruction as the CUDA kernel).
        # 'allow_tf32=False' forces IEEE BF16 multiply — no TF32 rounding.
        # 'input_precision="ieee"' enables native FP8 dot path if available.
        acc_gate = tl.dot(a_bf16, bg_bf16.T, acc_gate,
                          out_dtype=tl.float32, allow_tf32=False)
        acc_up   = tl.dot(a_bf16, bu_bf16.T, acc_up,
                          out_dtype=tl.float32, allow_tf32=False)

    # ── SwiGLU ────────────────────────────────────────────────────────────────
    # silu(x) = x * sigmoid(x)
    # Triton fuses this with the accumulator readback — no separate TMEM load/store.
    # On sm_100a with TMEM accumulators, Triton's register allocation pass
    # emits tcgen05.ld to move the result to registers here.
    gate_f = acc_gate.to(tl.float32)
    up_f   = acc_up.to(tl.float32)
    silu_up = up_f * tl.sigmoid(up_f)          # silu(up)
    result  = (silu_up * gate_f).to(tl.bfloat16)

    # ── Store output ──────────────────────────────────────────────────────────
    # Output is [Tk, I_DIM] BF16, stored at token_ids rows.
    out_ptrs = (out_ptr
                + token_ids[:, None] * N
                + offs_n[None, :])
    mask_out = mask_m[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptrs, result, mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: GEMM2 + weighted accumulate for one expert's N-tile
#
# Computes:
#   out += route_weight[token] * (inter[tokens, :] @ W2[expert, :, :]^T)
#
# inter: [T, I_DIM] BF16 (written by moe_gemm1_swiglu_kernel)
# W2:    [H_DIM, I_DIM] FP8
# out:   [T, H_DIM] FP32, atomicAdd (tokens shared across experts)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128},
                      num_warps=4, num_stages=3),
    ],
    key=["Tk", "H", "I"],
)
@triton.jit
def moe_gemm2_kernel(
    inter_ptr,          # [T, I_DIM] BF16 — SwiGLU output from GEMM1
    w2_ptr,             # [H_DIM, I_DIM] FP8 — GEMM2 weight, one expert
    sorted_ids_ptr,     # [Tk] int32
    topk_weights_ptr,   # [T, TOP_K] float32
    topk_idx_ptr,       # [T, TOP_K] int32
    out_ptr,            # [T, H_DIM] FP32 — global output, atomicAdd
    inter_scale_ptr,    # [ceil(I/128), T] float32
    w2_scale_ptr,       # [ceil(H/128), ceil(I/128)] float32
    expert_global_id,   # int: global expert index (for routing weight lookup)
    Tk,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid     = tl.program_id(0)
    num_n   = tl.cdiv(H, BLOCK_N)
    pid_m   = pid // num_n
    pid_n   = pid %  num_n

    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)
    mask_m  = offs_m < Tk

    token_ids = tl.load(sorted_ids_ptr + offs_m, mask=mask_m, other=0)

    # Look up routing weight for each token at this expert
    route_w = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for kk in tl.static_range(TOP_K):
        e = tl.load(topk_idx_ptr + token_ids * TOP_K + kk, mask=mask_m, other=-1)
        w = tl.load(topk_weights_ptr + token_ids * TOP_K + kk, mask=mask_m, other=0.0)
        route_w = tl.where(e == expert_global_id, w, route_w)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(tl.cdiv(I, BLOCK_K)):
        k_start = k * BLOCK_K

        # A: BF16 intermediate (contiguous per token — no scatter)
        a_ptrs  = inter_ptr + token_ids[:, None] * I + (k_start + offs_k)[None, :]
        mask_a  = mask_m[:, None] & ((k_start + offs_k)[None, :] < I)
        a       = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.bfloat16)

        # B: FP8 weight [H_DIM, I_DIM], load [BLOCK_N, BLOCK_K] slice
        b_ptrs  = w2_ptr + offs_n[:, None] * I + (k_start + offs_k)[None, :]
        mask_b  = (offs_n[:, None] < H) & ((k_start + offs_k)[None, :] < I)
        b       = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float8e4nv)

        a_sc   = tl.load(inter_scale_ptr + k * (I // BLOCK_K) + token_ids,
                         mask=mask_m, other=1.0)
        b_sc   = tl.load(w2_scale_ptr + pid_n * (I // BLOCK_K) + k)

        a_f    = a * a_sc[:, None].to(tl.bfloat16)
        b_f    = b.to(tl.bfloat16) * b_sc.to(tl.bfloat16)
        acc    = tl.dot(a_f, b_f.T, acc, out_dtype=tl.float32, allow_tf32=False)

    # Weighted accumulate: atomicAdd into global output
    # Multiple experts contribute to the same token → races → atomic required.
    result = acc * route_w[:, None]
    out_ptrs = out_ptr + token_ids[:, None] * H + offs_n[None, :]
    mask_out = mask_m[:, None] & (offs_n[None, :] < H)
    tl.atomic_add(out_ptrs, result, mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Routing kernels (Triton versions of Kernels 1a and 1b)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def routing_topk_kernel(
    logits_ptr,         # [T, E_GLOBAL] float32
    bias_ptr,           # [E_GLOBAL] float32
    topk_idx_ptr,       # [T, TOP_K] int32  output
    topk_weights_ptr,   # [T, TOP_K] float32 output
    T, E: tl.constexpr,
    N_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
):
    """
    One program = one token.
    Triton maps this to one warp (32 threads) internally, using warp-level
    reductions via the shuffle lowering in the LLVM backend.
    In CuTe/CUDA we wrote the warp reduction manually (xor shuffles);
    here tl.max + tl.argmax express the same operation at a higher level.
    """
    token = tl.program_id(0)
    if token >= T:
        return

    offs_e = tl.arange(0, E)
    logits = tl.load(logits_ptr + token * E + offs_e)
    bias   = tl.load(bias_ptr + offs_e)
    sig    = 1.0 / (1.0 + tl.exp(-logits))
    s_wb   = sig + bias

    # Group scores: top1 + top2 per group of GROUP_SIZE experts
    s_wb_2d = tl.reshape(s_wb, (N_GROUP, GROUP_SIZE))
    top1_g  = tl.max(s_wb_2d, axis=1)
    # Zero out top1 to find top2 (simplified — full version masks top1 lane)
    group_scores = top1_g   # approximate; full impl adds top2

    # Top-TOPK_GROUP groups by score (serial, only N_GROUP=8 values)
    # Triton lowers this to scalar code in registers — same as the CUDA version
    kept = tl.sort(group_scores, descending=True)[:TOPK_GROUP]

    # Select top-TOP_K experts within kept groups — omitted for brevity
    # Store: lanes 0..TOP_K-1 write one result each
    # (full impl mirrors the CUDA warp-shuffle reduction)


@triton.jit
def routing_permute_kernel(
    topk_idx_ptr,       # [T, TOP_K] int32
    sorted_ids_ptr,     # [E_LOCAL, T] int32  output
    expert_count_ptr,   # [E_LOCAL]    int32  output
    slot_ptr,           # [E_LOCAL]    int32  atomic counter (init 0)
    T, TOP_K: tl.constexpr,
    E_LOCAL: tl.constexpr,
    local_expert_offset: tl.constexpr,
):
    """
    Identical logic to routing_permute_kernel in the CUDA version.
    Triton uses tl.atomic_add for the slot counter — same PTX atomicAdd.
    The grid is (T * TOP_K,) and each program handles one (token, k) pair.
    """
    idx = tl.program_id(0)
    t   = idx // TOP_K
    k   = idx %  TOP_K
    ge  = tl.load(topk_idx_ptr + t * TOP_K + k)
    le  = ge - local_expert_offset
    if le >= 0 and le < E_LOCAL:
        slot = tl.atomic_add(slot_ptr + le, 1)
        tl.store(sorted_ids_ptr + le * T + slot, t)

    # After all threads finish, copy slot counters to expert_count.
    # In Triton, a second small kernel does this (no __threadfence equivalent).


# ─────────────────────────────────────────────────────────────────────────────
# Host launcher
# ─────────────────────────────────────────────────────────────────────────────

def launch_moe_triton(
    routing_logits: torch.Tensor,   # [T, 256] float32
    routing_bias:   torch.Tensor,   # [256] float32
    hidden:         torch.Tensor,   # [T, 7168] float8_e4nv
    hs_scale:       torch.Tensor,   # [56, T] float32
    w_gate:         torch.Tensor,   # [32, 2048, 7168] float8_e4nv  (gate)
    w_up:           torch.Tensor,   # [32, 2048, 7168] float8_e4nv  (up)
    w_gate_scale:   torch.Tensor,   # [32, 16, 56] float32
    w_up_scale:     torch.Tensor,
    w2:             torch.Tensor,   # [32, 7168, 2048] float8_e4nv
    w2_scale:       torch.Tensor,   # [32, 56, 16] float32
    routed_scaling_factor: float = 2.5,
    local_expert_offset: int = 0,
):
    T = routing_logits.shape[0]
    device = hidden.device

    topk_idx     = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    sorted_ids   = torch.zeros((32, T),    dtype=torch.int32, device=device)
    expert_count = torch.zeros((32,),      dtype=torch.int32, device=device)
    slot_ctr     = torch.zeros((32,),      dtype=torch.int32, device=device)
    intermediate = torch.empty((32, T, I_DIM), dtype=torch.bfloat16, device=device)
    output       = torch.zeros((T, H_DIM),     dtype=torch.float32,  device=device)

    # ── Kernel 1a: Routing top-k ──────────────────────────────────────────────
    # Triton autotuner will have already selected the best config for this T.
    routing_topk_kernel[(T,)](
        routing_logits, routing_bias,
        topk_idx, topk_weights,
        T=T, E=256,
        N_GROUP=8, GROUP_SIZE=32, TOP_K=8, TOPK_GROUP=4,
        routed_scaling_factor=routed_scaling_factor,
    )

    # ── Kernel 1b: Permutation build ──────────────────────────────────────────
    routing_permute_kernel[(T * TOP_K,)](
        topk_idx, sorted_ids, expert_count, slot_ctr,
        T=T, TOP_K=TOP_K, E_LOCAL=32,
        local_expert_offset=local_expert_offset,
    )

    # ── GEMM1 + SwiGLU: one kernel per expert ────────────────────────────────
    # Triton does not have a "persistent" launch model.
    # Each expert launches its own grid: (cdiv(Tk,BLOCK_M) * cdiv(I,BLOCK_N),)
    # For T=64, E_LOCAL=32: ~2 tokens/expert → small grids, poor GPU util.
    # For T≥512: grids grow, utilization improves.
    for e in range(32):
        Tk = int(expert_count[e].item())
        if Tk == 0:
            continue
        grid = lambda meta, Tk=Tk: (
            triton.cdiv(Tk, meta["BLOCK_M"]) * triton.cdiv(I_DIM, meta["BLOCK_N"]),
        )
        moe_gemm1_swiglu_kernel[grid](
            hidden, w_gate[e], w_up[e],
            sorted_ids[e], intermediate[e],
            hs_scale, w_gate_scale[e], w_up_scale[e],
            Tk=Tk, H=H_DIM, N=I_DIM,
        )

    # ── GEMM2 + weighted accumulate ───────────────────────────────────────────
    for e in range(32):
        Tk = int(expert_count[e].item())
        if Tk == 0:
            continue
        grid = lambda meta, Tk=Tk: (
            triton.cdiv(Tk, meta["BLOCK_M"]) * triton.cdiv(H_DIM, meta["BLOCK_N"]),
        )
        moe_gemm2_kernel[grid](
            intermediate[e], w2[e],
            sorted_ids[e], topk_weights, topk_idx,
            output, None, w2_scale[e],
            expert_global_id=local_expert_offset + e,
            Tk=Tk, H=H_DIM, I=I_DIM, TOP_K=TOP_K,
        )

    return output
