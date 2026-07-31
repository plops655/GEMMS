"""
modal_benchmark_moe.py — Run MoE Triton benchmarks on Modal with nsys profiling

Usage:
  modal run modal_benchmark_moe.py [--gpu H100|A100|L40S]

Outputs:
  - Persistent Modal volume: moe-benchmark-results
  - benchmark_results.json (timing data)
  - moe_triton_routing_topk.nsys-rep (GPU trace for routing_topk)
  - moe_triton_routing_permute.nsys-rep (GPU trace for routing_permute)
  - moe_triton_gemm1_swiglu.nsys-rep (GPU trace for gemm1_swiglu)
  - moe_triton_gemm2.nsys-rep (GPU trace for gemm2)

Download locally:
  modal volume get moe-benchmark-results ./local_results

View traces:
  nsys-ui local_results/moe_triton_routing_topk.nsys-rep
"""

import modal
import json
import subprocess
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Modal Setup
# ─────────────────────────────────────────────────────────────────────────────

# Use NVIDIA NGC container with nsys pre-installed
image = (
    modal.Image
    .from_registry("nvcr.io/nvidia/pytorch:24.07-py3")  # Has nsys + CUDA + Python
    .pip_install(
        "triton>=2.1.0",
    )
)

app = modal.App(
    name="moe-triton-benchmark",
    image=image,
)

# Persistent volume for results
results_volume = modal.Volume.from_name("moe-benchmark-results", create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Benchmark Function with Nsys Profiling
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu="H100",  # Can be overridden: A100, L40S, etc.
    timeout=3600,  # 60 minutes (profiling takes time)
    volumes={"/results": results_volume},
)
def run_benchmark_with_profiling():
    """
    Run MoE Triton benchmarks on Modal with full nsys profiling

    Benchmarks each kernel individually and collects GPU execution traces.
    Results and traces are saved to persistent volume.
    """
    import torch
    import time
    import sys
    import os

    results_dir = Path("/results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 90)
    print("MOE TRITON BENCHMARKS ON MODAL WITH NSYS PROFILING")
    print("=" * 90)

    # Check GPU
    print(f"\n📊 GPU Information:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"  Torch: {torch.__version__}")
    print(f"  Triton: {__import__('triton').__version__}")

    # Check nsys
    print(f"\n🔍 Profiling Tools:")
    result = subprocess.run(["which", "nsys"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ nsys: {result.stdout.strip()}")
    else:
        raise RuntimeError("nsys not found in container")

    # Import benchmark functions
    print(f"\n📦 Importing benchmark module...")
    try:
        # Try standard import first
        from benchmark_moe_triton import (
            BenchmarkConfig,
            benchmark_routing_topk,
            benchmark_routing_permute,
            benchmark_gemm1_swiglu,
            benchmark_gemm2,
        )
        print("  ✓ Successfully imported benchmark_moe_triton")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  Trying alternative import...")
        import importlib.util
        spec = importlib.util.find_spec("benchmark_moe_triton")
        if spec is None:
            raise RuntimeError(
                "Could not find benchmark_moe_triton. "
                "Make sure it's in the Modal working directory or mounted as a volume."
            )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BenchmarkConfig = mod.BenchmarkConfig
        benchmark_routing_topk = mod.benchmark_routing_topk
        benchmark_routing_permute = mod.benchmark_routing_permute
        benchmark_gemm1_swiglu = mod.benchmark_gemm1_swiglu
        benchmark_gemm2 = mod.benchmark_gemm2

    # Config for Modal (lighter: fewer repeats)
    config = BenchmarkConfig(
        num_tokens_list=[10, 100, 1000],  # Skip T=1 for profiling (too fast)
        num_experts=32,
        num_repeats=3,
        warmup_iters=2,
        device=torch.device("cuda"),
    )

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: Run benchmarks without profiling (quick)
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("STEP 1: RUNNING BENCHMARKS (Timing Data)")
    print("=" * 90)

    all_results = []
    all_results.extend(benchmark_routing_topk(config))
    all_results.extend(benchmark_routing_permute(config))
    all_results.extend(benchmark_gemm1_swiglu(config))
    all_results.extend(benchmark_gemm2(config))

    # Save timing results
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"\n✓ Timing results saved: {results_file}")


    # ────────────────────────────────────────────────────────────────────────
    # Step 3: Generate summaries
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("STEP 3: GENERATING SUMMARIES")
    print("=" * 90)

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: Profile each kernel with nsys
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("STEP 2: COLLECTING NSYS GPU TRACES")
    print("=" * 90)

    benchmarks = [
        ("routing_topk", benchmark_routing_topk),
        ("routing_permute", benchmark_routing_permute),
        ("gemm1_swiglu", benchmark_gemm1_swiglu),
        ("gemm2", benchmark_gemm2),
    ]

    for bench_name, bench_func in benchmarks:
        print(f"\n🔴 Profiling: {bench_name}")

        # Create wrapper script
        wrapper_script = results_dir / f"_bench_{bench_name}.py"
        with open(wrapper_script, "w") as f:
            f.write(f"""
import torch
from benchmark_moe_triton import BenchmarkConfig, {bench_func.__name__}

config = BenchmarkConfig(
    num_tokens_list=[100, 1000],
    num_experts=32,
    num_repeats=2,
    warmup_iters=1,
    device=torch.device("cuda"),
)

{bench_func.__name__}(config)
""")

        # Run nsys
        trace_file = results_dir / f"moe_triton_{bench_name}.nsys-rep"

        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            f"--output={trace_file}",
            "--force-overwrite=true",
            sys.executable,
            str(wrapper_script),
        ]

        print(f"   Running: nsys profile ...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=results_dir)

        if result.returncode == 0 and trace_file.exists():
            size_mb = trace_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ {trace_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ Failed: {result.stderr}")

    # Save summary
    summary = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(0),
        "timing_results": results_file.name,
    }
    summary_file = results_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary: {summary_file}")

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: List files
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("GENERATED FILES")
    print("=" * 90)

    total_size = 0
    for fpath in sorted(results_dir.glob("*")):
        if fpath.is_file() and not fpath.name.startswith("_"):
            size_mb = fpath.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{fpath.stat().st_size / 1024:.1f} KB"
            print(f"  {fpath.name:<50} {size_str:>10}")
            total_size += fpath.stat().st_size

    print(f"\nTotal size: {total_size / (1024 * 1024):.1f} MB")

    # ────────────────────────────────────────────────────────────────────────
    # Final instructions
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("✓ BENCHMARK COMPLETE")
    print("=" * 90)

    return {
        "status": "success",
        "gpu": torch.cuda.get_device_name(0),
        "results_saved_to_volume": "moe-benchmark-results",
        "download_command": "modal volume get moe-benchmark-results ./local_results",
        "view_traces": "nsys-ui local_results/moe_triton_*.nsys-rep",
        "stats": "nsys stats local_results/moe_triton_*.nsys-rep",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(gpu: str = "H100"):
    """
    Run benchmarks on Modal GPU with full nsys profiling

    Args:
        gpu: GPU type (H100, A100, L40S, etc.)

    Usage:
        modal run modal_benchmark_moe.py [--gpu H100]
    """
    import time

    print("=" * 90)
    print("LAUNCHING MOE TRITON BENCHMARK ON MODAL")
    print("=" * 90)
    print(f"\nGPU: {gpu}")
    print(f"Volume: moe-benchmark-results (persistent)")

    print("\n⏳ This will take 10-15 minutes (profiling overhead)...")
    print("   Running benchmarks and collecting GPU traces on Modal...\n")

    # Run benchmark on Modal
    result = run_benchmark_with_profiling.remote()

    print("\n" + "=" * 90)
    print("REMOTE EXECUTION COMPLETE")
    print("=" * 90)

    print("\n📋 Result Summary:")
    for key, value in result.items():
        if key != "status":
            print(f"  {key}: {value}")

    print("\n" + "=" * 90)
    print("NEXT STEPS")
    print("=" * 90)
    print(f"""
1️⃣  Download results from Modal:
    {result['download_command']}

2️⃣  View benchmark timing results:
    cat local_results/benchmark_results.json | jq .

3️⃣  View GPU trace in interactive GUI:
    {result['view_traces']}

4️⃣  Generate statistical report:
    {result['stats']}

5️⃣  Export to SQLite for custom analysis:
    nsys export -t sqlite local_results/moe_triton_routing_topk.nsys-rep

Happy profiling! 🚀
""")


if __name__ == "__main__":
    """Fallback for local testing"""
    print("This script is designed to run on Modal.")
    print("\nUsage:")
    print("  modal run modal_benchmark_moe.py [--gpu H100]")
    sys.exit(1)
