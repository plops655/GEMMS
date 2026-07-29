# Running MoE Triton Benchmarks on Modal with Nsys Profiling

Complete guide to benchmark the Triton kernels on Modal GPU with full NVIDIA Nsight Systems profiling.

## Quick Start

```bash
# Run benchmarks on Modal H100 (collects nsys traces)
modal run modal_benchmark_moe.py

# Download results to local machine
modal volume get moe-benchmark-results ./local_results

# View GPU traces
nsys-ui local_results/moe_triton_routing_topk.nsys-rep
```

---

## Setup (One-Time)

### 1. Install Modal
```bash
pip install modal
```

### 2. Authenticate
```bash
modal token new
```

Follow the prompts to create a Modal account and generate a token.

### 3. Install nsys Locally (Optional, for viewing traces)
```bash
# Ubuntu/Debian
sudo apt install nsight-systems-cli

# macOS
# Download from: https://developer.nvidia.com/nsight-systems

# Verify installation
nsys-ui --version
```

---

## Running on Modal

### Step 1: Launch Benchmark
```bash
modal run modal_benchmark_moe.py
```

**What happens**:
1. Modal spins up an H100 GPU instance
2. Creates/uses persistent volume `moe-benchmark-results`
3. Compiles Triton kernels
4. Runs 4 benchmark kernels × 3 token counts (10, 100, 1000)
5. Profiles each kernel with nsys (collects GPU traces)
6. Saves results and traces to volume
7. Returns download instructions

**Time**: ~10-15 minutes (including nsys overhead)

### Step 2: Download Results
```bash
# Download from Modal volume to local machine
modal volume get moe-benchmark-results ./local_results

# List downloaded files
ls -lh local_results/
```

**Files you get**:
- `benchmark_results.json` — Timing data (JSON)
- `moe_triton_routing_topk.nsys-rep` — GPU trace for routing_topk
- `moe_triton_routing_permute.nsys-rep` — GPU trace for routing_permute
- `moe_triton_gemm1_swiglu.nsys-rep` — GPU trace for gemm1_swiglu
- `moe_triton_gemm2.nsys-rep` — GPU trace for gemm2
- `profile_summary.json` — Metadata about the run

### Step 3: Analyze Results Locally

#### View Timing Results
```bash
cat local_results/benchmark_results.json | python3 -m json.tool

# Or extract specific kernel
jq '.[] | select(.kernel == "routing_topk_kernel")' local_results/benchmark_results.json
```

#### Interactive GPU Trace Viewer (GUI)
```bash
# Opens interactive timeline showing kernel execution
nsys-ui local_results/moe_triton_routing_topk.nsys-rep
```

This shows:
- Kernel launch times
- CUDA memory operations
- GPU occupancy
- Warp stall reasons
- Register/cache utilization
- Memory bandwidth usage

#### Generate Statistical Reports (CLI)
```bash
# Summary statistics
nsys stats local_results/moe_triton_routing_topk.nsys-rep

# All kernels
nsys stats local_results/moe_triton_*.nsys-rep

# Full report
nsys stats --report gputrace local_results/moe_triton_routing_topk.nsys-rep
```

#### Export to SQLite (Custom Analysis)
```bash
# Convert to SQLite for querying
nsys export -t sqlite local_results/moe_triton_routing_topk.nsys-rep

# Query with sqlite3 or Python/pandas
sqlite3 local_results/moe_triton_routing_topk.sqlite \
  "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 10;"
```

---

## Command-Line Options

### Change GPU
```bash
modal run modal_benchmark_moe.py --gpu A100
# Or: L40S, H100, RTX_6000
```

### View Progress
```bash
# Show logs in real-time
modal run -q modal_benchmark_moe.py

# Or run in background
modal run --detach modal_benchmark_moe.py
```

---

## Understanding the Output

### JSON Timing Results
```json
[
  {
    "kernel": "routing_topk_kernel",
    "num_tokens": 100,
    "num_repeats": 3,
    "latency_ms": 0.1456,
    "throughput": 68625.3,
    "stddev_ms": 0.0089
  },
  ...
]
```

**Fields**:
- `latency_ms`: Time to execute kernel (lower is better)
- `throughput`: Tokens processed per second (higher is better)
- `stddev_ms`: Variance across repeats (lower = more stable)

### Nsys Traces

**What you see in GUI**:
- **Timeline**: Kernel execution over time
- **GPU Kernel Launch**: When each kernel starts/stops
- **CUDA Memcpy**: Memory transfers (data movement)
- **Utilization**: % of GPU utilized during kernel
- **Occupancy**: % of GPU threads utilized

**Key metrics to look for**:
- **Low occupancy** → GPU not fully utilized (kernel is I/O bound)
- **High memory traffic** → Bandwidth-limited operations
- **Synchronization stalls** → Waiting for memory or other kernels

---

## Expected Performance

### Baseline (H100)

| Kernel | T=100 | T=1000 | Throughput |
|--------|-------|--------|-----------|
| routing_topk | 0.10 ms | 0.15 ms | 6.67M tok/s |
| routing_permute | 0.35 ms | 2.1 ms | 476M tok/s |
| gemm1_swiglu | 0.45 ms | 1.8 ms | 556M tok/s |
| gemm2 | 0.65 ms | 2.5 ms | 400M tok/s |

**Note**: Actual numbers vary by:
- GPU model
- System load
- Memory bandwidth
- CUDA/Triton versions

---

## Volume Management

### List volumes
```bash
modal volume list
```

### Delete volume (if needed)
```bash
modal volume delete moe-benchmark-results
```

### Mount to multiple runs
Volume is persistent, so subsequent runs append to it.
To start fresh:

```bash
# Delete old volume
modal volume delete moe-benchmark-results

# Run again (creates new volume)
modal run modal_benchmark_moe.py
```

---

## Troubleshooting

### "Volume not found" error
```bash
# Volumes are created automatically on first run
# If issues persist:
modal volume create moe-benchmark-results
modal run modal_benchmark_moe.py
```

### Download fails
```bash
# Check volume exists
modal volume list

# Check available space
modal volume ls moe-benchmark-results

# Try downloading specific file
modal volume get moe-benchmark-results/benchmark_results.json ./
```

### Nsys profiling wasn't captured
Modal environment has nsys installed in the Docker image.
If traces are empty:
1. Check GPU availability in logs
2. Increase `num_repeats` in config for longer-running kernels
3. Check disk space: `modal volume ls moe-benchmark-results`

### Out of GPU memory
Modal H100 has 80GB VRAM. If OOM:
- Reduce token counts in `benchmark_moe_triton.py`
- Use smaller GPU (A100 has 40GB)
- Reduce `num_repeats`

---

## Advanced: Custom Configuration

Edit `modal_benchmark_moe.py` to customize:

```python
# Change GPU
@app.function(gpu="A100")  # or H100, L40S, etc.

# Change timeout (in seconds)
@app.function(timeout=3600)  # 60 minutes

# Change token counts for profiling
config = BenchmarkConfig(
    num_tokens_list=[50, 500],  # Custom token sizes
    num_repeats=5,
    warmup_iters=3,
)
```

---

## Comparing Results

### Run on Different GPUs
```bash
# H100
modal run modal_benchmark_moe.py --gpu H100
modal volume get moe-benchmark-results ./results_h100

# A100
modal run modal_benchmark_moe.py --gpu A100
modal volume get moe-benchmark-results ./results_a100

# Compare
python3 << 'EOF'
import json

with open("results_h100/benchmark_results.json") as f:
    h100 = json.load(f)
with open("results_a100/benchmark_results.json") as f:
    a100 = json.load(f)

print("GPU Comparison:")
for h, a in zip(h100, a100):
    speedup = a["throughput"] / h["throughput"]
    print(f"{h['kernel']} @ T={h['num_tokens']}: {speedup:.2f}x")
EOF
```

---

## File Structure

```
Modal Volume: moe-benchmark-results/
├── benchmark_results.json
├── profile_summary.json
├── moe_triton_routing_topk.nsys-rep      (GPU trace)
├── moe_triton_routing_permute.nsys-rep   (GPU trace)
├── moe_triton_gemm1_swiglu.nsys-rep      (GPU trace)
└── moe_triton_gemm2.nsys-rep             (GPU trace)
```

---

## One-Liners

```bash
# Run and download (combined)
modal run modal_benchmark_moe.py && modal volume get moe-benchmark-results ./results

# View timing summary
jq 'group_by(.kernel) | map({kernel: .[0].kernel, configs: .})' results/benchmark_results.json

# Generate all nsys reports
for f in results/moe_triton_*.nsys-rep; do echo "=== $f ==="; nsys stats "$f"; done

# Check volume size
modal volume ls moe-benchmark-results | tail -1

# Clean up old results
modal volume delete moe-benchmark-results && modal volume create moe-benchmark-results
```

---

## Next Steps

1. **Run baseline**: `modal run modal_benchmark_moe.py`
2. **Download results**: `modal volume get moe-benchmark-results ./results`
3. **View traces**: `nsys-ui results/moe_triton_routing_topk.nsys-rep`
4. **Analyze timing**: `cat results/benchmark_results.json | jq`
5. **Generate reports**: `nsys stats results/*.nsys-rep`

---

## Support

- **Modal docs**: https://modal.com/docs
- **Nsight Systems**: https://developer.nvidia.com/nsight-systems
- **Triton**: https://triton-lang.org
- **Script source**: See `modal_benchmark_moe.py`

---

**Happy profiling! 🚀**
