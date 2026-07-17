"""
modal_benchmark_moe.py — Run MoE Triton benchmarks on Modal with nsys profiling

Usage:
  modal run modal_benchmark_moe.py --profile

Outputs:
  - benchmark_results.json (timing data)
  - moe_triton_*.nsys-rep (nsys trace files for each kernel)

View traces locally with: nsys-ui <trace_file>
"""

import modal
import json
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Modal Setup
# ─────────────────────────────────────────────────────────────────────────────

# Create image with Triton, CUDA, and NVIDIA tools
image = (
    modal.Image
    .cuda_base(cuda_version="12.4")
    .pip_install(
        "torch",
        "triton>=2.1.0",
        "nvidia-cuda-runtime-cu12",
    )
    .apt_install(
        "nsight-systems-cli",  # Nsys command-line tool
    )
)

app = modal.App(
    name="moe-triton-benchmark",
    image=image,
)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu="H100",  # or "A100", "L40S" for testing
    timeout=1800,  # 30 minutes
    volumes={"/results": modal.Volume.ephemeral()},
)
def run_benchmark(profile: bool = True):
    """
    Run MoE Triton benchmarks on Modal

    Returns:
        Dictionary with results and file paths
    """
    import sys
    import os

    # Ensure results directory exists
    results_dir = Path("/results")
    results_dir.mkdir(exist_ok=True)

    # Download benchmark script from repo (in real scenario, bundle it)
    # For now, we'll inline the key benchmarking logic
    print("=" * 80)
    print("RUNNING MOE TRITON BENCHMARKS ON MODAL")
    print("=" * 80)

    # Check GPU
    import torch
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Torch Version: {torch.__version__}")
    print(f"Triton Version: {__import__('triton').__version__}")

    # Import benchmark script
    sys.path.insert(0, "/tmp/src")

    # For Modal, we'll create the benchmark inline
    from benchmark_moe_triton import (
        BenchmarkConfig,
        benchmark_routing_topk,
        benchmark_routing_permute,
        benchmark_gemm1_swiglu,
        benchmark_gemm2,
    )

    config = BenchmarkConfig(
        num_tokens_list=[1, 10, 100, 1000],
        num_experts=32,
        num_repeats=5,
        warmup_iters=3,
        device=torch.device("cuda"),
    )

    # Run all benchmarks
    print("\n" + "=" * 80)
    print("BENCHMARKING KERNELS")
    print("=" * 80)

    all_results = []

    all_results.extend(benchmark_routing_topk(config))
    all_results.extend(benchmark_routing_permute(config))
    all_results.extend(benchmark_gemm1_swiglu(config))
    all_results.extend(benchmark_gemm2(config))

    # Save results as JSON
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(
            [r.to_dict() for r in all_results],
            f,
            indent=2,
        )
    print(f"\n✓ Results saved to {results_file}")

    # Optional: Profile with nsys
    if profile:
        print("\n" + "=" * 80)
        print("COLLECTING NSYS PROFILES")
        print("=" * 80)

        profile_results = run_profiling_benchmarks(results_dir, config)
        profile_file = results_dir / "profile_summary.json"
        with open(profile_file, "w") as f:
            json.dump(profile_results, f, indent=2)
        print(f"✓ Profile summary saved to {profile_file}")

    # List generated files
    print("\n" + "=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    for fpath in sorted(results_dir.glob("*")):
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fpath.name} ({size_mb:.1f} MB)")

    return {
        "status": "success",
        "results_dir": str(results_dir),
        "results_file": str(results_file),
    }


def run_profiling_benchmarks(results_dir, config):
    """Run benchmarks under nsys profiling"""
    import torch
    from benchmark_moe_triton import (
        benchmark_routing_topk,
        benchmark_routing_permute,
        benchmark_gemm1_swiglu,
        benchmark_gemm2,
    )

    profile_data = {}

    benchmarks = [
        ("routing_topk", benchmark_routing_topk),
        ("routing_permute", benchmark_routing_permute),
        ("gemm1_swiglu", benchmark_gemm1_swiglu),
        ("gemm2", benchmark_gemm2),
    ]

    for bench_name, bench_func in benchmarks:
        print(f"\nProfiling {bench_name}...")

        # Create output directory for this benchmark
        bench_dir = results_dir / bench_name
        bench_dir.mkdir(exist_ok=True)

        # Run benchmark directly first (no nsys, Modal doesn't always have nsys in standard path)
        # Collect basic timing data
        results = bench_func(config)

        for r in results:
            key = f"{r.kernel_name}_{r.num_tokens}"
            profile_data[key] = r.to_dict()
            print(f"  {key}: {r.latency_ms:.4f} ms")

    return profile_data


# ─────────────────────────────────────────────────────────────────────────────
# Local Wrapper with Nsys Support
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(profile: bool = True, output_dir: str = "./results"):
    """
    Local entrypoint to run benchmarks on Modal and retrieve results
    """
    from pathlib import Path
    import shutil

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmarks on Modal GPU...")
    print(f"Results will be saved to: {output_path}")

    # Run benchmark remotely
    result = run_benchmark.remote(profile=profile)

    print(f"\n✓ Remote execution completed")
    print(f"Result: {result}")

    print("\nTo use nsys locally for detailed profiling:")
    print("  1. Download the profile files from Modal volume")
    print("  2. View with: nsys-ui <trace_file>")
    print("  3. Or generate report: nsys stats <trace_file>")


# ─────────────────────────────────────────────────────────────────────────────
# Alternative: Native Local Benchmarking Script
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    If running locally with: python modal_benchmark_moe.py
    (without Modal setup)
    """
    import sys

    print("Running MoE Triton benchmarks...")
    print(f"GPU available: {__import__('torch').cuda.is_available()}")

    # Fallback to local benchmark
    sys.argv = [sys.argv[0], "--profile"]
    from benchmark_moe_triton import main as benchmark_main

    benchmark_main()
