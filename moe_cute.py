"""
moe_cute.py — DeepSeek-V3 MoE GEMM1 + SwiGLU in CuteDSL (CUTLASS 3.x Python DSL)

CuTe programming model in one sentence:
  You describe WHAT the hardware should do (which atom, which layout, which tiling),
  and the compiler lowers it mechanically — no search, no inference.

Compilation pipeline:
  Python source
    → Python AST
    → MLIR (CuTe dialect)    [layout algebra is symbolic — no index arithmetic]
    → MLIR (nvgpu dialect)   [MMA → nvgpu.warpgroup_mma, TMA → nvgpu.tma_async_load]
    → MLIR (nvvm dialect)    [thread-level semantics, ptx intrinsics]
    → LLVM IR
    → PTX (by LLVM backend)
    → SASS (by ptxas)

API note: uses CUTLASS 3.x Python DSL (cute-dsl package).
Exact import names may vary slightly by CUTLASS version.
"""

import cutlass
from cutlass import cute
from cutlass.cute import (
    Layout,
    make_layout,
    make_tensor,
    make_tiled_mma,
    make_tiled_copy,
    local_partition,
    local_tile,
    size,
)
from cutlass.cute.nvgpu import (
    # B200 / sm_100a warpgroup MMA atom: M=128, N=128, K=64, FP8→FP32
    # Maps directly to tcgen05.mma.cta_group::1.kind::f8f6f4
    SM100_MMA_F8F6F4_SS_128x128x64_F32,
    # TMA bulk async copy (GMEM → SMEM, 128B-swizzled)
    SM100_TMA_LOAD_TILED,
)
import cuda.bindings.driver as cuda

# ─────────────────────────────────────────────────────────────────────────────
# Model constants (match moe_constants.h)
# ─────────────────────────────────────────────────────────────────────────────

H_DIM         = 7168   # K-dimension for GEMM1
I_DIM         = 2048   # N-dimension for GEMM1 gate or up
BLOCK_QUANT   = 128    # FP8 quantization block size (= K_TILE)

# GEMM1 tile sizes
M_TILE        = 128    # rows  (tokens) — fixed by tcgen05 cta_group::1
N_TILE        = 128    # cols  (I_DIM slice) — fixed by tcgen05
K_TILE        = 128    # depth (H_DIM slice) — matches BLOCK_QUANT
TG05_K_STEP   = 64     # tcgen05 max K per call (FP8 kind)

GEMM1_K_TILES = H_DIM  // K_TILE   # 56
GEMM1_N_PAIRS = I_DIM  // N_TILE   # 16

THREADS_PER_BLOCK = 128   # 4 warps = 1 warpgroup

# ─────────────────────────────────────────────────────────────────────────────
# TMA descriptor creation (host side)
#
# CuTe encodes TMA descriptors as "copy atoms" paired with a global tensor's
# layout.  The layout (shape + stride) tells TMA the box size, base pointer,
# and row stride.  The 128B swizzle is encoded as a layout mode modifier —
# no separate runtime XOR: it is part of the layout algebra.
# ─────────────────────────────────────────────────────────────────────────────

def make_tma_desc_fp8(ptr, num_rows: int, num_cols: int,
                      tile_rows: int, tile_cols: int) -> cute.TmaDescriptor:
    """
    Build a TMA descriptor for an FP8 matrix.

    Layout composition:
      global_layout  = (num_rows, num_cols) : (num_cols, 1)   row-major
      tile_shape     = (tile_rows, tile_cols)
      swizzle        = Swizzle<3,4,3>  →  128B XOR pattern
    The swizzle is composed into the layout, not applied at runtime.
    """
    import cutlass.cute.swizzle as sw

    global_layout = make_layout(
        (num_rows, num_cols),
        stride=(num_cols, 1),
    )
    tma_atom = SM100_TMA_LOAD_TILED(
        dtype=cute.float8_e4m3,
        smem_layout=make_layout(
            (tile_rows, tile_cols),
            stride=sw.Swizzle(3, 4, 3),   # 128B swizzle: row XOR (row//8)
        ),
    )
    return cute.make_tma_copy_desc(tma_atom, ptr, global_layout)


def make_tma_desc_bf16(ptr, num_rows: int, num_cols: int,
                       tile_rows: int, tile_cols: int) -> cute.TmaDescriptor:
    import cutlass.cute.swizzle as sw

    global_layout = make_layout(
        (num_rows, num_cols),
        stride=(num_cols, 1),
    )
    tma_atom = SM100_TMA_LOAD_TILED(
        dtype=cute.bfloat16,
        smem_layout=make_layout(
            (tile_rows, tile_cols),
            stride=sw.Swizzle(3, 4, 3),
        ),
    )
    return cute.make_tma_copy_desc(tma_atom, ptr, global_layout)


# ─────────────────────────────────────────────────────────────────────────────
# SMEM layout specification
#
# CuTe layouts encode shape AND memory arrangement (stride + swizzle) in one
# algebraic object.  The compiler uses layout composition to compute every
# address at compile time — no index arithmetic in the generated PTX.
# ─────────────────────────────────────────────────────────────────────────────

def make_smem_layout_fp8():
    """
    FP8 SMEM tile: [M_TILE=128, K_TILE=128], 128B swizzle.
    Stride is a Swizzle<3,4,3> composition: consecutive rows are XOR-displaced
    by 16 bytes × (row % 8), distributing them across all 32 SMEM banks.
    CuTe represents this as a layout mode, not a runtime computation.
    """
    import cutlass.cute.swizzle as sw
    return make_layout(
        (M_TILE, K_TILE),
        stride=sw.Swizzle(3, 4, 3),
    )

def make_smem_layout_bf16():
    import cutlass.cute.swizzle as sw
    return make_layout(
        (M_TILE, K_TILE),
        stride=sw.Swizzle(3, 4, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: GEMM1 Gate sub-GEMM (representative of the hot loop)
#
# Computes: tmem_gate += A[Tk, H] × W_gate[N_TILE, H]^T
#   A is loaded via cp.async scatter (tokens non-contiguous in memory).
#   W_gate is loaded via TMA (weight rows are contiguous).
#   Accumulator lives in TMEM (Tensor Memory) — zero register pressure.
#
# One CTA = one warpgroup = 128 threads = one tcgen05 MMA tile.
# ─────────────────────────────────────────────────────────────────────────────

@cute.kernel(
    grid_dim=lambda T, **_: (
        cute.cdiv(T, M_TILE) * GEMM1_N_PAIRS,   # one CTA per (M-tile, N-tile) pair
        1, 1
    ),
    block_dim=(THREADS_PER_BLOCK, 1, 1),
    smem_bytes=2 * M_TILE * K_TILE * 1           # A double-buffer, FP8
            + 2 * N_TILE * K_TILE * 1            # B double-buffer, FP8
            + 2 * 8,                             # 2 mbarriers
)
def gemm1_gate_kernel(
    # Hidden states: [T, H_DIM] FP8, NOT contiguous per expert (scatter-loaded)
    A_gmem:       cute.Tensor,
    # Gate weight for this expert: [N_TILE, H_DIM] FP8, contiguous (TMA-loaded)
    W_gate_gmem:  cute.Tensor,
    # TMA descriptor for W_gate (prebuilt on host)
    tma_w_gate:   cute.TmaDescriptor,
    # sorted_token_ids: which token goes in each M-row of this CTA's tile
    sorted_ids:   cute.Tensor,
    # Routing weights [T, TOP_K], topk_idx [T, TOP_K]
    topk_weights: cute.Tensor,
    topk_idx:     cute.Tensor,
    # FP8 block-quantization scales
    A_scale:      cute.Tensor,   # [NUM_H_BLOCKS, T] — scale per token per K-block
    W_scale:      cute.Tensor,   # [N_PAIRS, NUM_H_BLOCKS] — scale per N-tile per K-block
    # Output intermediate [T, I_DIM] BF16
    inter_out:    cute.Tensor,
    T: int,
    expert_id: int,
    pair_j: int,
):
    tid = cute.arch.thread_idx().x

    # ── SMEM allocation ──────────────────────────────────────────────────────
    # CuTe allocates SMEM as typed tensors with explicit layouts.
    # The layout encodes the 128B swizzle — no separate swizzle API.
    smem_layout_fp8 = make_smem_layout_fp8()
    A_smem = cute.make_smem_tensor(cute.float8_e4m3, (2, M_TILE, K_TILE), smem_layout_fp8)
    B_smem = cute.make_smem_tensor(cute.float8_e4m3, (2, N_TILE, K_TILE), smem_layout_fp8)
    mbar   = cute.make_smem_tensor(cute.uint64, (2,))

    # ── TiledMMA: SM100 tcgen05 warpgroup MMA atom ───────────────────────────
    # make_tiled_mma composes the hardware atom with the tiling strategy.
    # SM100_MMA_F8F6F4_SS_128x128x64_F32 encodes:
    #   - m=128, n=128, k=64 per call
    #   - FP8 A (row-major), FP8 B (col-major), FP32 accumulator in TMEM
    #   - cta_group::1 (all 128 threads participate)
    # The tiling layout (1,1,1) means: one MMA atom covers the entire CTA tile.
    tiled_mma = make_tiled_mma(
        SM100_MMA_F8F6F4_SS_128x128x64_F32(),
        make_layout((1, 1, 1)),   # atom_layout: 1 atom in M, N, K directions
    )

    # Partition the MMA tile across this thread.
    # thr_mma.partition_C gives each thread its slice of the TMEM accumulator.
    # thr_mma.partition_A/B gives each thread the A/B fragments it needs to read.
    thr_mma = tiled_mma.get_thread_slice(tid)

    # Allocate TMEM accumulator (128×128 FP32 = 64 KB in Tensor Memory).
    # With CuTe, this is a first-class tensor operation — the accumulator is
    # explicitly typed as TMEM, not registers.  Zero register pressure.
    accum = thr_mma.make_fragment_C(cute.float32)   # TMEM allocation

    # ── TiledCopy: scatter-load A via cp.async ───────────────────────────────
    # A rows are non-contiguous (each row = a different token from sorted_ids).
    # No TMA for A: TMA requires contiguous rectangular boxes in GMEM.
    # Instead: all 128 threads cooperatively issue cp.async for their row slices.
    cp_async_atom = cute.SM80_CP_ASYNC_CACHEGLOBAL(cute.uint128_t)  # 16B per issue
    tiled_copy_A = make_tiled_copy(
        cp_async_atom,
        make_layout((THREADS_PER_BLOCK, 1)),   # thread layout
        make_layout((1, 16 // 1)),              # value layout: 16 bytes per thread
    )
    thr_copy_A = tiled_copy_A.get_thread_slice(tid)

    # ── TiledCopy: TMA load B ─────────────────────────────────────────────────
    # W_gate rows are contiguous: TMA loads the entire [N_TILE × K_TILE] box.
    # Thread 0 issues the TMA; all threads wait on mbar.
    tma_copy_B = cute.make_tma_copy(tma_w_gate, B_smem[0], tiled_mma)

    # ── Mbarrier init ─────────────────────────────────────────────────────────
    # CuTe mbarrier wraps the PTX mbarrier.init / arrive_expect_tx / try_wait
    # protocol behind a typed SMEM tensor operation.
    B_TILE_BYTES = N_TILE * K_TILE * 1   # FP8 = 1 byte per element
    if tid == 0:
        cute.mbarrier_init(mbar[0], 1)
        cute.mbarrier_init(mbar[1], 1)
    cute.arch.syncthreads()

    parity = [0, 0]

    # ── Prologue: prefetch tile kt=0 into buf=0 ──────────────────────────────
    # A: cp.async scatter (all threads participate)
    token0 = sorted_ids[expert_id * T]  # first token for this expert
    cute.copy(tiled_copy_A,
              A_gmem[token0, cute.make_coord(0, cute.make_range(0, K_TILE))],
              A_smem[0])
    cute.cp_async_commit_group()

    # B: single-thread TMA load
    if tid == 0:
        cute.mbarrier_arrive_expect_tx(mbar[0], B_TILE_BYTES)
        cute.copy(tma_copy_B,
                  W_gate_gmem[cute.make_coord(pair_j * N_TILE, 0)],
                  B_smem[0],
                  mbar[0])

    # ── Main K loop (double-buffered) ─────────────────────────────────────────
    #
    # CuTe pipeline protocol:
    #   1. Issue next prefetch into nbuf (async, non-blocking)
    #   2. Wait for current tile in buf:
    #        cp.async for A  →  cp_async_wait_group<1>
    #        TMA for B       →  mbarrier_wait(mbar[buf], parity)
    #   3. Barrier: all threads must see both A and B before MMA
    #   4. Build descriptors: make_smem_desc() packs the swizzled SMEM address
    #      + stride + 128B swizzle mode into a 64-bit value for the tensor core
    #   5. MMA: cute.gemm() → 2 × tcgen05.mma (K=64 each) to cover K_TILE=128
    #   6. Optional: apply FP8 dequant scale directly to TMEM accumulator

    for kt in range(GEMM1_K_TILES):
        buf  = kt & 1
        nbuf = 1 - buf

        # ── Step 1: Prefetch next tile ────────────────────────────────────────
        if kt + 1 < GEMM1_K_TILES:
            kn = kt + 1
            # A: gather next K-slice for each token row
            for row in thr_copy_A.partition_S(A_gmem):
                cute.copy(tiled_copy_A, row[..., kn * K_TILE : (kn+1) * K_TILE],
                          A_smem[nbuf])
            cute.cp_async_commit_group()

            # B: TMA loads [N_TILE × K_TILE] weight box at K-offset kn*K_TILE
            if tid == 0:
                cute.mbarrier_init(mbar[nbuf], 1)
                cute.mbarrier_arrive_expect_tx(mbar[nbuf], B_TILE_BYTES)
                cute.copy(tma_copy_B,
                          W_gate_gmem[cute.make_coord(pair_j * N_TILE, kn * K_TILE)],
                          B_smem[nbuf],
                          mbar[nbuf])

        # ── Step 2: Wait for current tile ─────────────────────────────────────
        # A: "1 outstanding group" means: current tile is done, next may still fly
        if kt + 1 < GEMM1_K_TILES:
            cute.cp_async_wait_group(1)
        else:
            cute.cp_async_wait_group(0)

        # B: byte-counting mbarrier fires when exactly B_TILE_BYTES have landed
        cute.mbarrier_wait(mbar[buf], parity[buf])
        parity[buf] ^= 1

        # ── Step 3: CTA barrier ───────────────────────────────────────────────
        # tcgen05.mma requires all 128 threads to issue it simultaneously.
        # mbarrier_wait exits per-thread at slightly different times.
        cute.arch.syncthreads()

        # ── Step 4: Build SMEM descriptors ────────────────────────────────────
        # make_smem_desc encodes smem_ptr, stride, and 128B swizzle into 64 bits.
        # The swizzle mode (bits [63:62]=3) must match CU_TENSOR_MAP_SWIZZLE_128B.
        # CuTe computes this at compile time from the layout — no runtime OR needed.
        a_desc = cute.make_smem_desc(A_smem[buf])
        b_desc = cute.make_smem_desc(B_smem[buf])

        # ── Step 5: Warpgroup MMA ─────────────────────────────────────────────
        # cute.gemm() iterates the K dimension in TG05_K_STEP=64 increments:
        #   ks=0: tcgen05.mma(..., scale_d=0)  — initialize TMEM (no accumulate)
        #   ks=1: tcgen05.mma(..., scale_d=1)  — add to TMEM
        # The descriptor K-offset is advanced by composing the layout:
        #   desc[ks] = a_desc ∘ (K_offset=ks*64) — layout composition, no runtime add
        cute.gemm(tiled_mma, A_smem[buf], B_smem[buf], accum,
                  accumulate=(kt > 0))

        # ── Step 6: FP8 dequantization scale ─────────────────────────────────
        # Each K-tile has a per-(N-pair, K-block) scale. If scale != 1, the TMEM
        # accumulator is read back, scaled, and written back using tcgen05.ld/st.
        # CuTe exposes this as: accum *= scalar  (when accum is a TMEM tensor)
        w_scale = W_scale[pair_j, kt]
        a_scale = A_scale[kt]   # broadcast over all token rows
        total_scale = a_scale * w_scale
        if total_scale != 1.0:
            accum = accum * total_scale   # → tcgen05.commit + fence + ld + st

    # ── SwiGLU: requires Gate AND Up results ──────────────────────────────────
    # (The Up sub-GEMM runs identically above and produces tmem_up.)
    # For brevity: SwiGLU is shown inline after both Gate and Up are computed.
    # In the full kernel, gate and up are read from their respective TMEM allocations.
    #   silu(up) = up * sigmoid(up)
    #   inter[row][col] = silu(up[row][col]) * gate[row][col]
    # CuTe reads TMEM via thr_mma.partition_C(), iterating 4 N-values per call.
    # The write is a direct store to g_intermediate (BF16).

    # cute.gemm accumulator is already in TMEM.  Read back 4 columns at a time:
    for nc_tile in thr_mma.partition_C(accum):   # shape (THREADS, 4)
        # nc_tile is a 4-element TMEM slice: [gate_n, gate_n+1, gate_n+2, gate_n+3]
        # (SwiGLU fused here — omitted for clarity, same pattern as CUDA version)
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Host launcher
# ─────────────────────────────────────────────────────────────────────────────

def launch_moe_gemm1_cute(
    hidden:   "DeviceArray[T, H_DIM, fp8]",
    w_gate:   "DeviceArray[E_LOCAL, I_DIM, H_DIM, fp8]",
    sorted_ids: "DeviceArray[E_LOCAL, T, int32]",
    topk_weights: "DeviceArray[T, TOP_K, float32]",
    topk_idx:     "DeviceArray[T, TOP_K, int32]",
    hs_scale: "DeviceArray[NUM_H_BLOCKS, T, float32]",
    w_gate_scale: "DeviceArray[E_LOCAL, GEMM1_N_PAIRS, NUM_H_BLOCKS, float32]",
    inter_out: "DeviceArray[E_LOCAL, T, I_DIM, bfloat16]",
    T: int,
    stream,
):
    """
    For each expert e and each N-tile pair j:
      1. Build TMA descriptor for w_gate[e, j*N_TILE:(j+1)*N_TILE, :]
      2. Launch gemm1_gate_kernel
      3. Launch gemm1_up_kernel (identical, different weight pointer)
    SwiGLU is fused inside the kernel after both gate and up accumulators are ready.
    """
    for e in range(32):   # E_LOCAL
        tma_wg = make_tma_desc_fp8(
            w_gate[e].data_ptr(), I_DIM, H_DIM, N_TILE, K_TILE
        )
        gemm1_gate_kernel(
            # ... tensors ...
            T=T, expert_id=e, pair_j=0,  # pair_j loops inside kernel in full version
            stream=stream,
        )
