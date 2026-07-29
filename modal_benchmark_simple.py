"""
modal_benchmark_simple.py — Simplified Modal runner for MoE Triton benchmarks

This version requires moe_triton.py and benchmark_moe_triton.py to be in a git repo.

Usage:
  # Option 1: Clone repo in Modal (recommended for dev)
  modal run modal_benchmark_simple.py --repo https://github.com/yourusername/flashinfer-comp

  # Option 2: Upload local files
  modal run modal_benchmark_simple.py --local-code

  # Option 3: Use installed modules (if pip-installed from git)
  modal run modal_benchmark_simple.py
"""

import modal
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Modal Setup
# ─────────────────────────────────────────────────────────────────────────────

image = (
    modal.Image
    .from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04")
    .pip_install(
        "torch",
        "triton>=2.1.0",
    )
    .apt_install(
        "nsight-systems-cli-remote",
        "curl",
        "wget",
        "git",
    )
)

app = modal.App(name="moe-triton-benchmark", image=image)

# Persistent volume
results_volume = modal.Volume.from_name("moe-benchmark-results", create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_benchmark(repo_url: str = None):
    """
    Run MoE Triton benchmarks with nsys profiling on Modal GPU
    """
    import torch
    import json
    import subprocess
    from pathlib import Path

    results_dir = Path("/results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 90)
    print("MOE TRITON BENCHMARKS ON MODAL")
    print("=" * 90)

    # GPU info
    print(f"\n📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"   Torch: {torch.__version__}")
    print(f"   Triton: {__import__('triton').__version__}")

    # Clone repo if provided
    if repo_url:
        print(f"\n📦 Cloning repository: {repo_url}")
        repo_dir = Path("/tmp/flashinfer")
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
        )
        import sys
        sys.path.insert(0, str(repo_dir))
    else:
        print("\n📦 Using installed modules (assuming moe_triton is available)")

    # Import benchmark functions
    print("\n⚙️  Loading benchmark module...")
    try:
        from benchmark_moe_triton import (
            BenchmarkConfig,
            benchmark_routing_topk,
            benchmark_routing_permute,
            benchmark_gemm1_swiglu,
            benchmark_gemm2,
        )
        print("   ✓ benchmark_moe_triton imported")
    except ImportError as e:
        print(f"   ✗ Error: {e}")
        print("\n💡 Solution:")
        print("   Use: modal run modal_benchmark_simple.py --repo <github-url>")
        print("   Or install: pip install git+https://github.com/yourusername/flashinfer-comp")
        raise

    # Run benchmarks
    print("\n" + "=" * 90)
    print("RUNNING BENCHMARKS")
    print("=" * 90)

    config = BenchmarkConfig(
        num_tokens_list=[10, 100, 1000],
        num_experts=32,
        num_repeats=3,
        warmup_iters=2,
        device=torch.device("cuda"),
    )

    all_results = []
    all_results.extend(benchmark_routing_topk(config))
    all_results.extend(benchmark_routing_permute(config))
    all_results.extend(benchmark_gemm1_swiglu(config))
    all_results.extend(benchmark_gemm2(config))

    # Save results
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"\n✓ Results: {results_file}")

    # Run nsys profiling
    print("\n" + "=" * 90)
    print("COLLECTING NSYS PROFILES")
    print("=" * 90)

    benchmarks = [
        ("routing_topk", benchmark_routing_topk),
        ("routing_permute", benchmark_routing_permute),
        ("gemm1_swiglu", benchmark_gemm1_swiglu),
        ("gemm2", benchmark_gemm2),
    ]

    for bench_name, bench_func in benchmarks:
        print(f"\n📌 Profiling: {bench_name}")

        # Create wrapper script
        wrapper = results_dir / f"_bench_{bench_name}.py"
        with open(wrapper, "w") as f:
            if repo_url:
                f.write("import sys; sys.path.insert(0, '/tmp/flashinfer')\n")
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
            "/usr/bin/python3",
            str(wrapper),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and trace_file.exists():
            size_mb = trace_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ Trace: {trace_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ Profile failed (continuing anyway)")

    # List files
    print("\n" + "=" * 90)
    print("FILES IN VOLUME")
    print("=" * 90)

    for fpath in sorted(results_dir.glob("*")):
        if not fpath.name.startswith("_"):
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  {fpath.name:<50} {size_mb:>8.1f} MB")

    print("\n" + "=" * 90)
    print("✓ DONE")
    print("=" * 90)

    return {"status": "success", "results_saved": "moe-benchmark-results"}


# ─────────────────────────────────────────────────────────────────────────────
# Local Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    repo: str = None,
    gpu: str = "H100",
):
    """
    Local entrypoint to run benchmarks on Modal

    Args:
        --repo: GitHub repo URL (e.g., https://github.com/user/flashinfer-comp)
        --gpu: GPU type (H100, A100, L40S)

    Usage:
        modal run modal_benchmark_simple.py --repo https://github.com/user/flashinfer-comp
    """
    print("=" * 90)
    print("LAUNCHING MOE TRITON BENCHMARK ON MODAL")
    print("=" * 90)
    print(f"\nGPU: {gpu}")
    print(f"Results volume: moe-benchmark-results")

    if not repo:
        print("\n⚠️  No repository provided.")
        print("\nUsage:")
        print("  modal run modal_benchmark_simple.py --repo <github-url>")
        print("\nExample:")
        print("  modal run modal_benchmark_simple.py --repo https://github.com/user/flashinfer-comp")
        print("\nAlternatively, install locally first:")
        print("  pip install git+https://github.com/user/flashinfer-comp")
        print("  modal run modal_benchmark_simple.py")
        return

    print("\n⏳ Running benchmark (~10 minutes)...\n")

    result = run_benchmark.remote(repo_url=repo)

    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)
    print(f"\n✓ Results saved to volume: {result['results_saved']}")
    print(f"\n📥 Download locally:")
    print(f"   modal volume get moe-benchmark-results ./local_results")
    print(f"\n📊 View timing results:")
    print(f"   cat local_results/benchmark_results.json | jq .")
    print(f"\n🔍 View GPU traces:")
    print(f"   nsys-ui local_results/moe_triton_routing_topk.nsys-rep")


if __name__ == "__main__":
    print("This script must be run with: modal run modal_benchmark_simple.py")
    sys.exit(1)
