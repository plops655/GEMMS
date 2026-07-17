# Quick Start: Benchmarking MoE Triton Kernels

## TL;DR

```bash
# Local benchmark (no profiling, ~2 minutes)
./run_benchmark.sh

# Local benchmark with nsys profiling (~10 minutes, requires NVIDIA tools)
./run_benchmark.sh results true

# On Modal cloud
modal run modal_benchmark_moe.py --profile
```

---

## Setup

### Prerequisites
```bash
pip install torch triton
```

### Optional: NVIDIA Profiling Tools
```bash
# Ubuntu/Debian
sudo apt install nsight-systems-cli

# macOS or manual install
# Download from: https://developer.nvidia.com/nsight-systems
```

---

## Running Locally

### Option 1: Simple Benchmark (Recommended for first run)
```bash
./run_benchmark.sh
```

**What it does**:
- Benchmarks 4 kernels (routing_topk, routing_permute, gemm1_swiglu, gemm2)
- Tests with 1, 10, 100, 1000 tokens
- Runs 5 repeats per configuration
- Outputs JSON with timing results

**Output**: `results/benchmark_results.json`

**Time**: ~2 minutes on H100

### Option 2: With Nsys Profiling
```bash
./run_benchmark.sh results true
```

**What it does**:
- Runs same benchmarks as Option 1
- Collects detailed GPU execution traces
- Generates nsys profile files

**Output**:
- `results/benchmark_results.json` — timing data
- `results/*.nsys-rep` — GPU traces

**Time**: ~10 minutes on H100

### Option 3: Custom Parameters
```bash
./run_benchmark.sh <output_dir> <profile> <num_repeats> <warmup_iters>

# Example: More repeats, more warmup, no profiling
./run_benchmark.sh my_results false 10 5
```

---

## Analyzing Results

### View Timing Summary
```bash
cat results/benchmark_results.json | python3 -m json.tool
```

### Plot Results (Python)
```python
import json
import matplotlib.pyplot as plt

with open("results/benchmark_results.json") as f:
    data = json.load(f)

kernels = set(r["kernel"] for r in data)
for kernel in kernels:
    results = [r for r in data if r["kernel"] == kernel]
    tokens = [r["num_tokens"] for r in results]
    latency = [r["latency_ms"] for r in results]
    
    plt.plot(tokens, latency, marker="o", label=kernel)

plt.xlabel("Token Count")
plt.ylabel("Latency (ms)")
plt.legend()
plt.xscale("log")
plt.savefig("benchmark_results.png")
```

### View GPU Traces (if profiling enabled)
```bash
# GUI viewer
nsys-ui results/moe_triton_routing_topk_profile.nsys-rep

# CLI statistics
nsys stats results/moe_triton_routing_topk_profile.nsys-rep

# Export to SQLite for custom analysis
nsys export -t sqlite results/moe_triton_routing_topk_profile.nsys-rep
```

---

## Running on Modal

### Setup Modal
```bash
pip install modal
modal token new  # Authenticate
```

### Run Benchmark
```bash
modal run modal_benchmark_moe.py --profile
```

### Retrieve Results
Default: results are ephemeral

To persist:
```python
# Edit modal_benchmark_moe.py
volumes={"/results": modal.Volume.from_name("moe-results", create_if_missing=True)}
```

Then download:
```bash
modal volume get moe-results ./local_results
```

---

## Understanding the Output

### JSON Structure
```json
[
  {
    "kernel": "routing_topk_kernel",
    "num_tokens": 1,
    "num_repeats": 5,
    "latency_ms": 0.1234,           // Lower is better
    "throughput": 8064.5,           // Higher is better (tokens/sec)
    "stddev_ms": 0.0042             // Lower is better (stability)
  },
  ...
]
```

### Interpreting Latency
- **0.1 ms** = 100 microseconds (very fast, likely I/O bound)
- **1.0 ms** = 1 millisecond (typical for medium compute)
- **10 ms** = 10 milliseconds (heavy compute or small GPU utilization)

### Interpreting Throughput
- **1M tokens/sec** = processes 1M tokens per second
- **1G tokens/sec** = processes 1 billion tokens per second

For T=1000:
- Latency 1 ms → throughput = 1,000 / 0.001s = 1M tokens/sec ✓

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce repeats and warmup
./run_benchmark.sh results false 1 1
```

### "nsys not found" but profiling enabled
```bash
# Skip profiling
./run_benchmark.sh results false
```

### High variance in results
```bash
# Increase warmup and repeats for more stable numbers
./run_benchmark.sh results false 10 10
```

### Triton compilation timeout
First run compiles kernels. Subsequent runs use cache:
```bash
# Run once without benchmark to compile
python3 -c "from moe_triton import *"

# Then run benchmark
./run_benchmark.sh
```

---

## File Organization

```
flashinfer-comp/
├── benchmark_moe_triton.py      # Main benchmark script
├── modal_benchmark_moe.py        # Modal cloud wrapper
├── run_benchmark.sh              # Helper shell script
├── BENCHMARK_README.md           # Detailed documentation
├── BENCHMARK_QUICKSTART.md       # This file
├── moe_triton.py                 # Triton kernels (imported)
└── results/                      # Output directory
    ├── benchmark_results.json    # Timing results
    └── *.nsys-rep               # GPU traces (if profiling enabled)
```

---

## Common Workflows

### Baseline Measurement
```bash
./run_benchmark.sh baseline_h100 true
```

### Regression Testing
```bash
./run_benchmark.sh results_new false 3 2
# Compare with: diff results_old/benchmark_results.json results_new/benchmark_results.json
```

### Kernel Tuning
1. Run baseline: `./run_benchmark.sh baseline false`
2. Modify kernel (e.g., BLOCK_M, BLOCK_N)
3. Run new: `./run_benchmark.sh tuned false`
4. Compare results

### Multi-GPU Comparison
```bash
# On GPU 0 (H100)
CUDA_VISIBLE_DEVICES=0 ./run_benchmark.sh results_gpu0 true

# On GPU 1 (A100)
CUDA_VISIBLE_DEVICES=1 ./run_benchmark.sh results_gpu1 true

# Compare:
python3 -c "
import json
with open('results_gpu0/benchmark_results.json') as f: h100 = json.load(f)
with open('results_gpu1/benchmark_results.json') as f: a100 = json.load(f)
for h, a in zip(h100, a100):
    print(f\"{h['kernel']} @ T={h['num_tokens']}: {a['throughput']/h['throughput']:.2f}x\")
"
```

---

## Performance Expectations (H100)

| Kernel | T=10 | T=100 | T=1000 |
|--------|------|-------|--------|
| routing_topk | 0.08 ms | 0.10 ms | 0.15 ms |
| routing_permute | 0.12 ms | 0.35 ms | 2.1 ms |
| gemm1_swiglu | 0.15 ms | 0.45 ms | 1.8 ms |
| gemm2 | 0.20 ms | 0.65 ms | 2.5 ms |

**Actual numbers depend on**: GPU model, CUDA version, system load, memory bandwidth.

---

## Next Steps

1. **Run baseline**: `./run_benchmark.sh`
2. **Review results**: `cat results/benchmark_results.json`
3. **Profile specific kernel**: `nsys-ui results/*.nsys-rep`
4. **Tweak kernel**: Edit `BLOCK_M`, `BLOCK_N` in `moe_triton.py`
5. **Re-benchmark**: `./run_benchmark.sh`
6. **Compare**: Check if new params are faster

---

## Help

- **Benchmark details**: See `BENCHMARK_README.md`
- **Kernel source**: See `moe_triton.py` (lines 69-309)
- **Triton docs**: https://triton-lang.org
- **Nsight Systems**: https://developer.nvidia.com/nsight-systems

---

## One-Liners

```bash
# Quick check (30 seconds)
python3 benchmark_moe_triton.py --num-repeats 1 --warmup-iters 1

# Full benchmark with profiling
./run_benchmark.sh my_results true 10 5

# Compare two results
diff <(jq 'sort_by(.kernel,.num_tokens)' results1/benchmark_results.json) \
     <(jq 'sort_by(.kernel,.num_tokens)' results2/benchmark_results.json)

# Extract routing_topk results only
jq '.[] | select(.kernel == "routing_topk_kernel")' results/benchmark_results.json

# Find slowest configuration
jq 'max_by(.latency_ms)' results/benchmark_results.json

# Find best throughput
jq 'max_by(.throughput)' results/benchmark_results.json
```

---

**Happy benchmarking! 🚀**
