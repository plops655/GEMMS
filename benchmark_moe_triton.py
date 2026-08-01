"""
benchmark_moe_triton.py — Benchmark individual Triton MoE kernels

Usage:
  Local:  python benchmark_moe_triton.py --profile [--output-dir ./results]
  Modal:  modal run benchmark_moe_triton.py --profile
"""

import torch
import triton
import triton.language as tl
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
import json
import subprocess
import sys

# Import MoE kernels
from moe_triton import (
    moe_gemm1_swiglu_kernel,
    moe_gemm2_kernel,
    routing_topk_kernel,
    routing_permute_kernel,
    H_DIM,
    I_DIM,
    TOP_K,
    BLOCK_QUANT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    num_tokens_list: list = None  # [1, 10, 100, 1000]
    num_experts: int = 32
    num_repeats: int = 5
    warmup_iters: int = 3
    device: torch.device = None

    def __post_init__(self):
        if self.num_tokens_list is None:
            self.num_tokens_list = [1, 10, 100, 1000]
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    kernel_name: str
    num_tokens: int
    num_repeats: int
    latency_ms: float  # mean latency per iteration (ms)
    throughput: float  # tokens/sec or tokens processed
    stddev_ms: float = 0.0

    def to_dict(self):
        return {
            "kernel": self.kernel_name,
            "num_tokens": self.num_tokens,
            "num_repeats": self.num_repeats,
            "latency_ms": round(self.latency_ms, 4),
            "throughput": round(self.throughput, 2),
            "stddev_ms": round(self.stddev_ms, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dummy Input Generation
# ─────────────────────────────────────────────────────────────────────────────

def create_routing_inputs(T, device):
    """Create inputs for routing kernels"""
    routing_logits = torch.randn(T, 256, dtype=torch.float32, device=device)
    routing_bias = torch.randn(256, dtype=torch.float32, device=device)
    topk_idx = torch.empty(T, TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(T, TOP_K, dtype=torch.float32, device=device)

    return routing_logits, routing_bias, topk_idx, topk_weights


def create_gemm1_inputs(Tk, E_local, device):
    """Create inputs for GEMM1 kernel"""
    hidden = torch.randn(Tk, H_DIM, dtype=torch.float16, device=device)
    w_gate = torch.randn(E_local, I_DIM * 2, H_DIM, dtype=torch.float16, device=device)
    w_up = torch.randn(E_local, I_DIM * 2, H_DIM, dtype=torch.float16, device=device)
    sorted_ids = torch.arange(Tk, dtype=torch.int32, device=device)

    intermediate = torch.empty(Tk, I_DIM, dtype=torch.bfloat16, device=device)

    # Scales
    num_hidden_blocks = H_DIM // BLOCK_QUANT  # 56
    num_intermediate_blocks = I_DIM // BLOCK_QUANT  # 16
    num_gemm1_out_blocks = (2 * I_DIM) // BLOCK_QUANT  # 32

    hs_scale = torch.ones(num_hidden_blocks, Tk, dtype=torch.float32, device=device)
    w_gate_scale = torch.ones(E_local, num_gemm1_out_blocks, num_hidden_blocks,
                               dtype=torch.float32, device=device)
    w_up_scale = torch.ones(E_local, num_gemm1_out_blocks, num_hidden_blocks,
                             dtype=torch.float32, device=device)

    return (hidden, w_gate, w_up, sorted_ids, intermediate,
            hs_scale, w_gate_scale, w_up_scale)


def create_gemm2_inputs(Tk, E_local, device):
    """Create inputs for GEMM2 kernel"""
    inter = torch.randn(Tk, I_DIM, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(E_local, H_DIM, I_DIM, dtype=torch.float16, device=device)
    sorted_ids = torch.arange(Tk, dtype=torch.int32, device=device)

    topk_weights = torch.ones(Tk, TOP_K, dtype=torch.float32, device=device) / TOP_K
    topk_idx = torch.zeros(Tk, TOP_K, dtype=torch.int32, device=device)
    # Set one expert per token (for sparse testing)
    topk_idx[:, 0] = torch.randint(0, E_local, (Tk,), device=device)

    out = torch.zeros(Tk, H_DIM, dtype=torch.float32, device=device)

    # Scales
    num_hidden_blocks = H_DIM // BLOCK_QUANT  # 56
    num_intermediate_blocks = I_DIM // BLOCK_QUANT  # 16

    inter_scale = torch.ones(num_intermediate_blocks, Tk, dtype=torch.float32, device=device)
    w2_scale = torch.ones(E_local, num_hidden_blocks, num_intermediate_blocks,
                          dtype=torch.float32, device=device)

    return inter, w2, sorted_ids, topk_weights, topk_idx, out, inter_scale, w2_scale


def create_permute_inputs(T, device):
    """Create inputs for permute kernel"""
    topk_idx = torch.randint(0, 256, (T, TOP_K), dtype=torch.int32, device=device)
    sorted_ids = torch.zeros(32, T, dtype=torch.int32, device=device)
    expert_count = torch.zeros(32, dtype=torch.int32, device=device)
    slot_ptr = torch.zeros(32, dtype=torch.int32, device=device)

    return topk_idx, sorted_ids, expert_count, slot_ptr


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking Functions
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_function(func_name, func, args, config, profile=False):
    """
    Benchmark a single function across different batch sizes

    Args:
        func_name: Name of kernel
        func: Callable kernel function
        args: Callable that returns args given num_tokens (or similar)
        config: BenchmarkConfig
        profile: Whether to profile with nsys

    Returns:
        List of BenchmarkResult
    """
    results = []
    device = config.device

    print(f"\n{'='*70}")
    print(f"Benchmarking: {func_name}")
    print(f"{'='*70}")

    for num_tokens in config.num_tokens_list:
        print(f"\n  Testing with {num_tokens} tokens...")

        # Create inputs
        kernel_args = args(num_tokens, device)
        if not isinstance(kernel_args, (list, tuple)):
            kernel_args = [kernel_args]

        # Warmup
        print(f"    Warming up ({config.warmup_iters} iterations)...", end="", flush=True)
        for _ in range(config.warmup_iters):
            func(*kernel_args)
        torch.cuda.synchronize(device)
        print(" ✓")

        # Benchmark
        print(f"    Benchmarking ({config.num_repeats} repeats)...", end="", flush=True)

        times = []
        for _ in range(config.num_repeats):
            # Recreate inputs to avoid caching effects
            kernel_args = args(num_tokens, device)
            if not isinstance(kernel_args, (list, tuple)):
                kernel_args = [kernel_args]

            torch.cuda.synchronize(device)
            start = time.perf_counter()

            func(*kernel_args)

            torch.cuda.synchronize(device)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        print(" ✓")

        # Compute statistics
        latency_ms = sum(times) / len(times)
        stddev_ms = (sum((t - latency_ms) ** 2 for t in times) / len(times)) ** 0.5
        throughput = (num_tokens * config.num_repeats) / (sum(times) / 1000)  # tokens/sec

        result = BenchmarkResult(
            kernel_name=func_name,
            num_tokens=num_tokens,
            num_repeats=config.num_repeats,
            latency_ms=latency_ms,
            throughput=throughput,
            stddev_ms=stddev_ms,
        )
        results.append(result)

        print(f"    Latency: {latency_ms:.4f} ± {stddev_ms:.4f} ms")
        print(f"    Throughput: {throughput:.0f} tokens/sec")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_routing_topk(config):
    """Benchmark routing top-k kernel"""
    def args_fn(T, device):
        routing_logits, routing_bias, topk_idx, topk_weights = create_routing_inputs(T, device)
        return [
            routing_logits, routing_bias,
            topk_idx, topk_weights,
            T, 256,
            8, 32, TOP_K, 4,
            2.5,  # routed_scaling_factor
        ]

    def wrapper(*args):
        routing_topk_kernel[(args[4],)](  # T,
            args[0], args[1],  # logits_ptr, bias_ptr
            args[2], args[3],  # topk_idx_ptr, topk_weights_ptr
            args[4], args[5],  # T, E
            args[6], args[7], args[8], args[9],  # N_GROUP, GROUP_SIZE, TOP_K, TOPK_GROUP
            args[10],  # routed_scaling_factor
        )

    return benchmark_function("routing_topk_kernel", wrapper, args_fn, config)


def benchmark_routing_permute(config):
    """Benchmark routing permute kernel"""
    def args_fn(T, device):
        topk_idx, sorted_ids, expert_count, slot_ptr = create_permute_inputs(T, device)
        return [topk_idx, sorted_ids, expert_count, slot_ptr, T]

    def wrapper(*args):
        routing_permute_kernel[(args[4] * TOP_K,)](  # (T * TOP_K,)
            args[0], args[1], args[2], args[3],  # topk_idx_ptr, sorted_ids_ptr, expert_count_ptr, slot_ptr
            args[4], TOP_K, 32, 0,  # T, TOP_K, E_LOCAL, local_expert_offset
        )

    return benchmark_function("routing_permute_kernel", wrapper, args_fn, config)


def benchmark_gemm1_swiglu(config):
    """Benchmark GEMM1 + SwiGLU kernel"""
    def args_fn(T, device):
        hidden, w_gate, w_up, sorted_ids, intermediate, hs_scale, w_gate_scale, w_up_scale = \
            create_gemm1_inputs(T, config.num_experts, device)

        # Use first expert
        return [
            hidden, w_gate[0], w_up[0], sorted_ids, intermediate,
            hs_scale, w_gate_scale[0], w_up_scale[0],
            T, H_DIM, I_DIM * 2,  # Tk, H, N
        ]

    def wrapper(*args):
        hidden_ptr, w_gate_ptr, w_up_ptr, sorted_ids_ptr, out_ptr = args[:5]
        hs_scale_ptr, wg_scale_ptr, wu_scale_ptr = args[5:8]
        Tk, H, N = args[8:11]

        grid = lambda meta, Tk=Tk: (
            triton.cdiv(Tk, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

        moe_gemm1_swiglu_kernel[grid](
            hidden_ptr, w_gate_ptr, w_up_ptr, sorted_ids_ptr, out_ptr,
            hs_scale_ptr, wg_scale_ptr, wu_scale_ptr,
            Tk=Tk, H=H, N=N,
        )

    return benchmark_function("moe_gemm1_swiglu_kernel", wrapper, args_fn, config)


def benchmark_gemm2(config):
    """Benchmark GEMM2 + weighted accumulate kernel"""
    def args_fn(T, device):
        inter, w2, sorted_ids, topk_weights, topk_idx, out, inter_scale, w2_scale = \
            create_gemm2_inputs(T, config.num_experts, device)

        # Use first expert
        return [
            inter, w2[0], sorted_ids, topk_weights, topk_idx, out,
            inter_scale, w2_scale[0],
            0,  # expert_global_id
            T, H_DIM, I_DIM, TOP_K,  # Tk, H, I, TOP_K
        ]

    def wrapper(*args):
        inter_ptr, w2_ptr, sorted_ids_ptr, topk_weights_ptr, topk_idx_ptr = args[:5]
        out_ptr, inter_scale_ptr, w2_scale_ptr = args[5:8]
        expert_global_id = args[8]
        Tk, H, I, TOP_K_val = args[9:13]

        grid = lambda meta, Tk=Tk: (
            triton.cdiv(Tk, meta["BLOCK_M"]) * triton.cdiv(H, meta["BLOCK_N"]),
        )

        moe_gemm2_kernel[grid](
            inter_ptr, w2_ptr, sorted_ids_ptr, topk_weights_ptr, topk_idx_ptr,
            out_ptr, inter_scale_ptr, w2_scale_ptr,
            expert_global_id=expert_global_id,
            Tk=Tk, H=H, I=I, TOP_K=TOP_K_val,
        )

    return benchmark_function("moe_gemm2_kernel", wrapper, args_fn, config)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton MoE kernels")
    parser.add_argument("--profile", action="store_true", help="Profile with nsys")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num-repeats", type=int, default=5, help="Number of benchmark repeats")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Warmup iterations")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")

    # Create config
    config = BenchmarkConfig(
        num_tokens_list=[1, 10, 100, 1000],
        num_experts=32,
        num_repeats=args.num_repeats,
        warmup_iters=args.warmup_iters,
        device=device,
    )

    # Collect all results
    all_results = []

    # Run benchmarks
    if args.profile:
        print("\n⚠️  Profiling mode: Running under nsys...")
        # This will be handled by wrapper script
        benchmark_with_profiling(config, output_dir)
    else:
        all_results.extend(benchmark_routing_topk(config))
        all_results.extend(benchmark_routing_permute(config))
        all_results.extend(benchmark_gemm1_swiglu(config))
        all_results.extend(benchmark_gemm2(config))

        # Save results
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(
                [r.to_dict() for r in all_results],
                f,
                indent=2,
            )
        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for kernel_name in ["routing_topk_kernel", "routing_permute_kernel",
                           "moe_gemm1_swiglu_kernel", "moe_gemm2_kernel"]:
            print(f"\n{kernel_name}:")
            kernel_results = [r for r in all_results if r.kernel_name == kernel_name]
            for r in kernel_results:
                print(f"  T={r.num_tokens:4d}: {r.latency_ms:.4f} ms "
                      f"({r.throughput:.0f} tokens/sec)")


def benchmark_with_profiling(config, output_dir):
    """Run benchmarks under nsys profiling"""
    print("Running benchmarks with nsys profiling...")

    trace_file = output_dir / "moe_triton_profile.nsys-rep"

    # Run each benchmark with profiling
    benchmarks = [
        ("routing_topk", benchmark_routing_topk),
        ("routing_permute", benchmark_routing_permute),
        ("gemm1_swiglu", benchmark_gemm1_swiglu),
        ("gemm2", benchmark_gemm2),
    ]

    for bench_name, bench_func in benchmarks:
        print(f"\nProfiling {bench_name}...")

        # Create a wrapper script that calls the benchmark
        wrapper_script = output_dir / f"run_{bench_name}.py"
        with open(wrapper_script, "w") as f:
            f.write(f"""
import sys
sys.path.insert(0, "{Path(__file__).parent}")
from benchmark_moe_triton import {bench_func.__name__}, BenchmarkConfig
import torch

config = BenchmarkConfig(
    num_tokens_list=[10, 100, 1000],
    num_experts=32,
    num_repeats=3,
    warmup_iters=2,
    device=torch.device("cuda"),
)

results = {bench_func.__name__}(config)
for r in results:
    print(r.to_dict())
""")

        # Run with nsys
        nsys_output = output_dir / f"{bench_name}_profile.nsys-rep"
        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={nsys_output}",
            "--force-overwrite=true",
            sys.executable, str(wrapper_script),
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=output_dir)

        if result.returncode == 0:
            print(f"  ✓ Profile saved to {nsys_output}")
        else:
            print(f"  ✗ Profiling failed")


if __name__ == "__main__":
    main()
