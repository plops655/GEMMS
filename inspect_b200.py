"""
inspect_b200.py — Run on a B200 to:
  1. Verify Triton emits tcgen05 (not wgmma fallback)
  2. Dump IR at every compilation stage
  3. Compare against the hand-written CUDA PTX

Usage:
    python inspect_b200.py             # local if you have a B200
    modal run inspect_b200.py          # remote via Modal
"""

import os
import re
import sys

# ── Modal entry point (wrap the whole thing) ──────────────────────────────────
try:
    import modal
    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False

if _HAS_MODAL:
    from pathlib import Path
    app = modal.App("b200-triton-inspect")
    image = (
        modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3")
        .pip_install("triton")
        .add_local_dir(
            str(Path(__file__).parent),
            remote_path="/src",
            copy=True,
            ignore=lambda p: not (str(p).endswith(".py") or str(p).endswith(".h")),
        )
    )

    @app.function(image=image, gpu="B200", timeout=300)
    def run_remote():
        os.chdir("/src")
        results = run_inspection()
        return results

    @app.local_entrypoint()
    def main():
        results = run_remote.remote()
        for fname, content in results.items():
            with open(fname, "w") as f:
                f.write(content)
            print(f"Saved {fname}  ({len(content):,} chars)")


# ── Core inspection (runs on GPU machine) ─────────────────────────────────────

def run_inspection():
    import torch
    import triton
    import triton.language as tl

    # ── 1. Verify GPU ─────────────────────────────────────────────────────────
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"\nGPU: {name}")
    print(f"Compute capability: {cap[0]}.{cap[1]}")
    # B200 = 10.0; H100 = 9.0
    is_b200 = (cap == (10, 0))
    print(f"Is B200 (sm_100a): {is_b200}")
    if not is_b200:
        print("WARNING: not a B200 — Triton will use wgmma (sm_90) instead of tcgen05 (sm_100a)")

    # ── 2. Define a minimal FP8 GEMM kernel (same tile as hot loop) ───────────
    # BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 mirrors one tcgen05.mma call.
    # This is the smallest kernel that forces Triton to pick the MMA atom.

    @triton.jit
    def fp8_gemm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in tl.range(tl.cdiv(K, BLOCK_K)):
            a = tl.load(A_ptr + offs_m[:, None] * stride_am
                                + (k * BLOCK_K + offs_k)[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & ((k * BLOCK_K + offs_k)[None, :] < K),
                        other=0.0).to(tl.float8e4nv)
            b = tl.load(B_ptr + (k * BLOCK_K + offs_k)[:, None] * stride_bk
                                + offs_n[None, :] * stride_bn,
                        mask=((k * BLOCK_K + offs_k)[:, None] < K) & (offs_n[None, :] < N),
                        other=0.0).to(tl.float8e4nv)
            # tl.dot: Triton's F32DotTCPass decides which MMA atom to emit.
            # On sm_100a: should pick tcgen05.mma.cta_group::1.kind::f8f6f4
            # On sm_90:   would pick wgmma.mma_async.sync.aligned
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="ieee")

        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.bfloat16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # ── 3. Compile with num_stages=2 (forces PipelinePass double-buffer) ──────
    M, N, K = 128, 128, 7168   # one GEMM1 tile: 128 tokens × 7168 H_DIM
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

    # Use triton.compile() directly — gives us the full .asm dict
    compiled = triton.compile(
        fp8_gemm_kernel,
        signature={
            "A_ptr": "*fp8e4nv", "B_ptr": "*fp8e4nv", "C_ptr": "*bf16",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
        },
        constants={
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        },
        num_warps=4,
        num_stages=2,   # PipelinePass will double-buffer the K loop
    )

    results = {}

    # Save each IR stage
    for stage in ["ttir", "ttgir", "llir", "ptx"]:
        src = compiled.asm.get(stage, "")
        if src:
            fname = f"triton_gemm1_{stage}.txt"
            results[fname] = src
            print(f"\n{'='*60}")
            print(f"Stage: {stage.upper()}  ({len(src):,} chars)")
            print(f"{'='*60}")

            if stage == "ptx":
                analyze_ptx(src)
            elif stage == "ttgir":
                analyze_ttgir(src)

    # ── 4. Warmup + benchmark ─────────────────────────────────────────────────
    # Allocate as float8 using torch — torch.float8_e4m3fn = tl.float8e4nv
    dtype_fp8 = torch.float8_e4m3fn
    A = torch.randn(M, K, device="cuda").to(dtype_fp8)
    B = torch.randn(K, N, device="cuda").to(dtype_fp8)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    grid = (1, 1)
    for _ in range(3):  # warmup
        fp8_gemm_kernel[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
    torch.cuda.synchronize()

    import time
    N_BENCH = 100
    start = time.perf_counter()
    for _ in range(N_BENCH):
        fp8_gemm_kernel[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / N_BENCH

    flops = 2 * M * N * K  # multiply-add per element
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    print(f"\nBenchmark: {elapsed_ms:.3f} ms/call  →  {tflops:.1f} TFLOPS")
    results["benchmark.txt"] = f"{elapsed_ms:.3f} ms  {tflops:.1f} TFLOPS\n"

    return results


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_ptx(ptx: str):
    """Scan PTX for the key instructions and print a summary."""

    checks = {
        # B200-native: should see these on sm_100a
        "tcgen05.mma":            "B200 warpgroup MMA (tcgen05) ← GOOD on B200",
        "tcgen05.alloc":          "TMEM allocation",
        "tcgen05.dealloc":        "TMEM deallocation",
        "tcgen05.ld":             "TMEM readback",
        "tcgen05.commit":         "TMEM commit/fence",
        "cp.async.bulk.tensor":   "TMA bulk load (B200/Hopper)",
        "mbarrier.arrive.expect": "Byte-counting mbarrier (pipeline sync)",
        "mbarrier.try_wait":      "Mbarrier poll (pipeline sync)",

        # Hopper fallback: should NOT see these if truly on sm_100a path
        "wgmma.mma_async":        "Hopper wgmma ← FALLBACK, not using tcgen05",

        # Ampere-era: should not appear in the hot loop
        "mma.sync.aligned.m16n8": "Ampere-era per-warp MMA ← old path",
        "cp.async.cg.shared":     "Ampere-era cp.async (ok for scatter A load)",
    }

    print("\nPTX instruction analysis:")
    for instr, label in checks.items():
        count = ptx.count(instr)
        marker = "✓" if count > 0 else "·"
        if "FALLBACK" in label and count > 0:
            marker = "✗"
        if "GOOD" in label and count == 0:
            marker = "✗  MISSING"
        print(f"  {marker} {count:3d}×  {instr:38s}  {label}")

    # Register count from .reg directive
    reg_match = re.search(r'\.reg \.b32\s+%r<(\d+)>', ptx)
    if reg_match:
        print(f"\n  Register pressure: {reg_match.group(1)} × b32 regs per thread")

    # Check .target line — must be sm_100a for tcgen05 to be legal
    target_match = re.search(r'\.target\s+(sm_\w+)', ptx)
    if target_match:
        target = target_match.group(1)
        ok = target == "sm_100a"
        print(f"\n  .target: {target}  {'← correct for B200' if ok else '← WRONG: need sm_100a'}")


def analyze_ttgir(ttgir: str):
    """Show key TritonGPU IR features — evidence of which passes fired."""

    print("\nTritonGPU IR pass evidence:")

    # PipelinePass: inserts async copy + mbarrier around the K loop
    pipeline_markers = [
        ("triton_nvidia_gpu.async_tma_copy_global_to_local",
         "PipelinePass → TMA async copy inserted"),
        ("triton_nvidia_gpu.mbarrier_arrive_expect_tx",
         "PipelinePass → mbarrier byte-count signaling"),
        ("triton_nvidia_gpu.mbarrier_wait",
         "PipelinePass → mbarrier wait (tiles double-buffered)"),
        ("triton_gpu.local_alloc",
         "SharedMemoryAllocationPass → SMEM buffers allocated"),
        ("triton_gpu.memdesc_subslice",
         "PipelinePass → double-buffer slice indexing"),
        ("#triton_nvidia_gpu.tensor_memory_encoding",
         "tcgen05 TMEM accumulator encoding (WarpSpecialization or F32DotTC)"),
        ("triton_gpu.warp_specialize",
         "WarpSpecializationPass → producer/consumer warp split"),
    ]

    for marker, label in pipeline_markers:
        count = ttgir.count(marker)
        sym = "✓" if count > 0 else "·"
        print(f"  {sym} {count:3d}×  {label}")

    # Show which MMA encoding was assigned to the dot op
    dot_enc = re.search(r'tt\.dot.*?#(triton_\w+\.\w+)', ttgir)
    if dot_enc:
        print(f"\n  tl.dot encoding: {dot_enc.group(1)}")


# ── Standalone entry (not Modal) ──────────────────────────────────────────────

if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "modal"):
    results = run_inspection()
    for fname, content in results.items():
        with open(fname, "w") as f:
            f.write(content)
    print(f"\nSaved {list(results.keys())}")
