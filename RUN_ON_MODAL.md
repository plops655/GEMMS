# Running Benchmarks on Modal - Complete Guide

The easiest way to run your benchmarks on Modal with nsys profiling:

## Option 1: Using Git Repository (Recommended)

### Step 1: Push your code to GitHub
```bash
cd /Users/jayanthsadhasivan/Desktop/flashinfer-comp

# If not already a git repo
git init
git add .
git commit -m "Add MoE benchmarking"

# Create repo on GitHub and push
git remote add origin https://github.com/yourusername/flashinfer-comp
git push -u origin main
```

### Step 2: Run on Modal
```bash
modal run modal_benchmark_simple.py --repo https://github.com/yourusername/flashinfer-comp
```

### Step 3: Download results
```bash
modal volume get moe-benchmark-results ./local_results
```

### Step 4: View results
```bash
# Timing data
cat local_results/benchmark_results.json | jq .

# GPU traces
nsys-ui local_results/moe_triton_routing_topk.nsys-rep
```

---

## Option 2: Using Current Directory (Simpler)

If you don't want to use git:

### Step 1: Make sure Modal can access your files

Create a wrapper script that copies files:

```bash
cat > /tmp/run_on_modal.sh << 'EOF'
#!/bin/bash

cd /Users/jayanthsadhasivan/Desktop/flashinfer-comp

# Create a git repo temporarily
git init --initial-branch=main
git add .
git commit -m "benchmark" 2>/dev/null || true

# Get the absolute path
REPO_PATH=$(pwd)

# Run Modal with the local path
modal run modal_benchmark_simple.py --repo file://$REPO_PATH

# Clean up
rm -rf .git
EOF

chmod +x /tmp/run_on_modal.sh
/tmp/run_on_modal.sh
```

Or simply:

### Step 1: Create minimal Modal script

Copy the flashinfer-comp directory to Modal's context:

```bash
cd /Users/jayanthsadhasivan/Desktop/flashinfer-comp
modal run modal_benchmark_simple.py
```

This won't work without files. Instead, use the git approach above.

---

## Recommended Workflow

### Quick Setup (5 minutes)

```bash
cd /Users/jayanthsadhasivan/Desktop/flashinfer-comp

# 1. Create GitHub repo (if not existing)
git init
git add .
git commit -m "MoE Triton benchmarks with nsys profiling"
git remote add origin https://github.com/yourusername/flashinfer-comp
git push -u origin main

# 2. Run on Modal H100 (15 minutes)
modal run modal_benchmark_simple.py --repo https://github.com/yourusername/flashinfer-comp

# 3. Download results (2 minutes)
modal volume get moe-benchmark-results ./results_h100

# 4. Analyze (local machine)
cat results_h100/benchmark_results.json | jq .
nsys-ui results_h100/moe_triton_routing_topk.nsys-rep
```

**Total time**: ~25 minutes

---

## Installation Prerequisites

Make sure Modal is installed:

```bash
pip install modal

# Authenticate
modal token new
```

---

## Understanding the Output

### benchmark_results.json
```json
[
  {
    "kernel": "routing_topk_kernel",
    "num_tokens": 100,
    "latency_ms": 0.1234,
    "throughput": 68625.3,
    "stddev_ms": 0.0089
  }
]
```

### nsys Traces
Interactive GPU execution timelines showing:
- Kernel launches
- Memory operations
- GPU occupancy
- Warp stalls

---

## Common Issues

### "ImportError: No module named 'benchmark_moe_triton'"
**Solution**: Use `--repo` flag to specify GitHub URL

```bash
modal run modal_benchmark_simple.py --repo https://github.com/yourusername/flashinfer-comp
```

### "Volume not found"
**Solution**: Modal creates it automatically on first run. If issues persist:

```bash
modal volume create moe-benchmark-results
modal run modal_benchmark_simple.py --repo <your-repo>
```

### "GPU out of memory"
**Solution**: Edit `modal_benchmark_simple.py`, change token counts:

```python
config = BenchmarkConfig(
    num_tokens_list=[10, 100],  # Reduced from [10, 100, 1000]
    num_repeats=2,  # Reduced from 3
)
```

### Download hangs
**Solution**: Check volume size first:

```bash
modal volume ls moe-benchmark-results

# If huge, delete and re-run
modal volume delete moe-benchmark-results
modal run modal_benchmark_simple.py --repo <your-repo>
```

---

## Monitoring Progress

### Watch logs in real-time
```bash
modal run modal_benchmark_simple.py --repo <your-repo>
# or
modal run -q modal_benchmark_simple.py --repo <your-repo>
```

### Check volume usage
```bash
modal volume ls moe-benchmark-results
```

### List all Modal runs
```bash
modal app list
```

---

## Advanced: Change GPU Type

Edit `modal_benchmark_simple.py`:

```python
@app.function(
    gpu="A100",  # Change here: H100, A100, L40S, etc.
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_benchmark(repo_url: str = None):
```

Then run:

```bash
modal run modal_benchmark_simple.py --repo <your-repo>
```

---

## Expected Times

| Step | Time |
|------|------|
| Benchmark execution | 3-5 min |
| Nsys profiling | 5-10 min |
| Download | 1-2 min |
| Total | ~15 min |

---

## Next Steps

1. **Push to GitHub**: Get your repo URL ready
2. **Run benchmark**: Use the command above
3. **Download results**: `modal volume get moe-benchmark-results ./results`
4. **Analyze locally**:
   - View timing: `cat results/benchmark_results.json | jq .`
   - View traces: `nsys-ui results/moe_triton_*.nsys-rep`
   - Generate reports: `nsys stats results/moe_triton_*.nsys-rep`

---

## One-Liner Quick Start

```bash
# Assumes you have a GitHub repo already
modal run modal_benchmark_simple.py --repo https://github.com/yourusername/flashinfer-comp && \
  modal volume get moe-benchmark-results ./results && \
  cat results/benchmark_results.json | jq '.[0:2]'
```

---

**Ready? Let's go! 🚀**

```bash
modal run modal_benchmark_simple.py --repo https://github.com/yourusername/flashinfer-comp
```
