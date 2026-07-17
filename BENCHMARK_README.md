# MoE Triton Kernel Benchmarking & Profiling

Complete benchmarking suite for DeepSeek MoE Triton kernels with NVIDIA Nsight Systems integration.

## Quick Start

### Local Benchmarking (No Profiling)
```bash
python benchmark_moe_triton.py
```

Output: `./results/benchmark_results.json`

### Local Benchmarking with Nsys Profiling
```bash
python benchmark_moe_triton.py --profile
```

Output:
- `./results/benchmark_results.json` — timing data (JSON)
- `./results/moe_triton_*.nsys-rep` — nsys trace files (one per kernel)

### Remote Benchmarking on Modal
```bash
modal run modal_benchmark_moe.py --profile
```

Runs on Modal GPU cluster, collects results and timing data.

---

## Detailed Usage

### Command-Line Options

```bash
python benchmark_moe_triton.py \
  --profile              # Enable nsys profiling (requires NVIDIA tools)
  --output-dir ./results # Output directory (default: ./results)
  --num-repeats 5        # Repeats per config (default: 5)
  --warmup-iters 3       # Warmup iterations (default: 3)
```

### What Gets Benchmarked?

| Kernel | Input Shape | Tested Token Counts |
|--------|-------------|-------------------|
| `routing_topk_kernel` | [T, 256] | 1, 10, 100, 1000 |
| `routing_permute_kernel` | [T, TOP_K=8] | 1, 10, 100, 1000 |
| `moe_gemm1_swiglu_kernel` | [T, 7168] → [T, 2048] | 1, 10, 100, 1000 |
| `moe_gemm2_kernel` | [T, 2048] → [T, 7168] | 1, 10, 100, 1000 |

For each kernel × token count:
- **Warmup**: 3 iterations (default)
- **Benchmark**: 5 repeats (default), measure latency & throughput

---

## Output Format

### JSON Results (`benchmark_results.json`)

```json
[
  {
    "kernel": "routing_topk_kernel",
    "num_tokens": 1,
    "num_repeats": 5,
    "latency_ms": 0.1234,
    "throughput": 8064.5,
    "stddev_ms": 0.0042
  },
  {
    "kernel": "routing_topk_kernel",
    "num_tokens": 10,
    "num_repeats": 5,
    "latency_ms": 0.1456,
    "throughput": 68625.3,
    "stddev_ms": 0.0089
  },
  ...
]
```

**Fields**:
- `kernel`: Kernel name
- `num_tokens`: Number of tokens in batch
- `num_repeats`: Number of benchmark iterations
- `latency_ms`: Mean latency (milliseconds)
- `throughput`: Tokens processed per second
- `stddev_ms`: Standard deviation of latency across repeats

### Nsys Trace Files

Each `*.nsys-rep` file contains a full GPU execution trace with:
- CUDA kernel launches
- Memory operations
- Synchronization points
- Timeline of execution

---

## Analyzing Results with Nsight Systems

### View Trace File (GUI)
```bash
nsys-ui results/moe_triton_routing_topk_profile.nsys-rep
```

Opens interactive timeline viewer showing:
- Kernel launch times
- Memory bandwidth utilization
- GPU occupancy
- Warp stall reasons

### Generate Report (CLI)
```bash
nsys stats results/moe_triton_routing_topk_profile.nsys-rep
```

Outputs summary statistics:
```
CUDA Kernel Statistics:
  Kernel Name          | Calls | Total Time | Avg Time | Max Time
  ───────────────────────────────────────────────────────────────
  routing_topk_kernel  |  15   | 2.345 ms   | 0.156 ms | 0.189 ms
```

### Export to SQLite (Advanced)
```bash
nsys export -t sqlite results/moe_triton_routing_topk_profile.nsys-rep
```

Query with Python/pandas for custom analysis.

---

## Understanding the Metrics

### Latency (ms)
Time to execute one kernel call (average across repeats).

```
Lower is better.
Triton kernels typically: 0.1–2.0 ms depending on token count.
```

**Example interpretation**:
- `routing_topk` with T=1000: 0.15 ms = 150 microseconds
- `gemm1_swiglu` with T=1000: 1.2 ms = 1200 microseconds

### Throughput (tokens/sec)
Tokens processed per second.

```
throughput = (num_tokens * num_repeats) / total_time_seconds

Higher is better.
```

**Example interpretation**:
- `routing_topk` with T=1000: 6.67M tokens/sec
  - Processes 1000 tokens in 0.15 ms
  - Amortized over 5 repeats

### Stddev (ms)
Standard deviation of latency across repeats.

```
Lower is better (more stable execution).
Typical range: 0.001–0.05 ms for well-tuned kernels.

High stddev can indicate:
  - Kernel contention (other processes)
  - Uneven occupancy
  - Variable-latency operations (atomic adds)
```

---

## Performance Expectations

### Baseline Numbers (H100, FP8 quantization)

| Kernel | T=10 Latency | T=100 Latency | T=1000 Latency | T=1000 Throughput |
|--------|--------------|---------------|----------------|-------------------|
| routing_topk | 0.08 ms | 0.10 ms | 0.15 ms | 6.67M tokens/sec |
| routing_permute | 0.12 ms | 0.35 ms | 2.1 ms | 476M tokens/sec |
| gemm1_swiglu | 0.15 ms | 0.45 ms | 1.8 ms | 556M tokens/sec |
| gemm2 | 0.20 ms | 0.65 ms | 2.5 ms | 400M tokens/sec |

**Note**: These are estimates; actual numbers depend on:
- GPU model (H100, A100, L40S, etc.)
- CUDA version
- Triton version
- System load
- Memory bandwidth availability

---

## Troubleshooting

### "CUDA out of memory"
Reduce `num_tokens` or decrease `--num-repeats`.

```bash
python benchmark_moe_triton.py --num-repeats 2
```

### "Nsys not found"
Install NVIDIA tools:
```bash
# Ubuntu
sudo apt install nsight-systems-cli

# macOS (if available)
# Or download from: https://developer.nvidia.com/nsight-systems
```

### "Triton compilation slow on first run"
Expected behavior. Triton JIT-compiles each kernel → first run caches result.

**Solution**: Run once without benchmarking to warmup:
```bash
python -c "from benchmark_moe_triton import *; print('Triton initialized')"
```

### High variance in measurements
Possible causes:
1. **System load**: Close other applications
2. **Dynamic frequency scaling**: Disable CPU power management
3. **L2 cache thrashing**: Increase `--warmup-iters`

```bash
# Disable dynamic frequency (Linux, requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

python benchmark_moe_triton.py --warmup-iters 10
```

---

## Modal Cloud Deployment

### Prerequisites
```bash
pip install modal
modal token new
```

### Run on Modal H100
```bash
modal run modal_benchmark_moe.py --profile
```

### Retrieve Results
Modal volumes are ephemeral by default. To persist results:

Edit `modal_benchmark_moe.py`:
```python
volumes={"/results": modal.Volume.from_name("moe-results", create_if_missing=True)},
```

Then download locally:
```bash
modal volume get moe-results ./local_results
```

### Specify Different GPU
```python
@app.function(gpu="A100")  # or "L40S", "H100"
```

---

## Advanced: Custom Benchmarks

### Add Your Own Kernel

```python
def benchmark_my_kernel(config):
    """Benchmark a custom kernel"""
    def args_fn(T, device):
        # Create dummy inputs
        return [tensor1, tensor2, ...]
    
    def wrapper(*args):
        # Call your kernel
        my_kernel(*args)
    
    return benchmark_function("my_kernel", wrapper, args_fn, config)


# In main():
all_results.extend(benchmark_my_kernel(config))
```

### Change Token Counts
```python
config.num_tokens_list = [8, 32, 64, 256, 512]
```

### Compare GPU Models
Run on different GPUs and compare JSON files:

```bash
python benchmark_moe_triton.py --output-dir ./results_h100
modal run modal_benchmark_moe.py --profile  # (set gpu="A100" in script)
```

Then analyze:
```python
import json

with open("results_h100/benchmark_results.json") as f:
    h100_results = json.load(f)

with open("results_a100/benchmark_results.json") as f:
    a100_results = json.load(f)

# Compare throughputs
for h100, a100 in zip(h100_results, a100_results):
    speedup = a100["throughput"] / h100["throughput"]
    print(f"{h100['kernel']} @ T={h100['num_tokens']}: {speedup:.2f}x speedup")
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `benchmark_moe_triton.py` | Main benchmarking script (local) |
| `modal_benchmark_moe.py` | Modal cloud deployment wrapper |
| `moe_triton.py` | Triton kernel implementations (imported) |
| `results/benchmark_results.json` | Timing results (JSON) |
| `results/moe_triton_*.nsys-rep` | Nsys trace files (binary) |

---

## Performance Tuning Tips

1. **Warmup thoroughly**: More iterations → more stable numbers
   ```bash
   --warmup-iters 10
   ```

2. **Run multiple repeats**: Smooths out noise
   ```bash
   --num-repeats 10
   ```

3. **Pin GPU frequency** (if available):
   ```bash
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -lgc 1980  # Lock GPU to 1980 MHz
   ```

4. **Monitor power/thermal**:
   ```bash
   nvidia-smi -l 1  # Refresh every second
   ```

5. **Profile specific configurations**: Run once with full profiling, then use smaller repeats for iteration.

---

## Citation

If using these benchmarks in publications:

```bibtex
@misc{flashinfer_moe_triton,
  title={DeepSeek MoE Triton Kernel Benchmarks},
  author={Flashinfer Contributors},
  year={2024},
  url={https://github.com/flashinfer-ai/flashinfer}
}
```

---

## Questions?

- Check kernel source in `moe_triton.py`
- Review Triton docs: https://triton-lang.org
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
