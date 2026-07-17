#!/bin/bash
# run_benchmark.sh — Benchmark runner with nsys integration

set -e

# Configuration
OUTPUT_DIR="${1:-.results}"
PROFILE="${2:-false}"
NUM_REPEATS="${3:-5}"
WARMUP="${4:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Check prerequisites
check_requirements() {
    print_info "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"

    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "NVIDIA CUDA not found"
        exit 1
    fi
    print_success "NVIDIA CUDA found: $(nvidia-smi --version | head -1)"

    # Check for nsys if profiling
    if [ "$PROFILE" = "true" ]; then
        if ! command -v nsys &> /dev/null; then
            print_error "nsys (NVIDIA Nsight Systems) not found"
            print_info "Install with: sudo apt install nsight-systems-cli"
            print_info "Or disable profiling: run_benchmark.sh <dir> false"
            exit 1
        fi
        print_success "nsys found: $(nsys --version 2>&1 | grep -i version || echo 'nsys CLI')"
    fi

    # Check GPU
    print_success "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
}

# Create output directory
setup_output() {
    mkdir -p "$OUTPUT_DIR"
    print_success "Output directory: $(cd "$OUTPUT_DIR" && pwd)"
}

# Run benchmark
run_benchmark() {
    print_header "RUNNING BENCHMARK"

    cd "$SCRIPT_DIR"

    if [ "$PROFILE" = "true" ]; then
        print_info "Profiling enabled - this will take longer"
        python3 benchmark_moe_triton.py \
            --profile \
            --output-dir "$OUTPUT_DIR" \
            --num-repeats "$NUM_REPEATS" \
            --warmup-iters "$WARMUP"
    else
        print_info "Profiling disabled (faster)"
        python3 benchmark_moe_triton.py \
            --output-dir "$OUTPUT_DIR" \
            --num-repeats "$NUM_REPEATS" \
            --warmup-iters "$WARMUP"
    fi
}

# Analyze results
analyze_results() {
    print_header "RESULTS ANALYSIS"

    RESULTS_FILE="$OUTPUT_DIR/benchmark_results.json"

    if [ ! -f "$RESULTS_FILE" ]; then
        print_error "Results file not found: $RESULTS_FILE"
        return 1
    fi

    print_success "Results file: $RESULTS_FILE"

    # Print summary table
    echo ""
    python3 << 'EOF'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        results = json.load(f)
except:
    sys.exit(1)

# Group by kernel
kernels = {}
for r in results:
    k = r['kernel']
    if k not in kernels:
        kernels[k] = []
    kernels[k].append(r)

# Print table
print("BENCHMARK SUMMARY")
print("─" * 90)
print(f"{'Kernel':<30} {'Tokens':<10} {'Latency (ms)':<20} {'Throughput':<15}")
print("─" * 90)

for kernel_name in sorted(kernels.keys()):
    for r in kernels[kernel_name]:
        lat_str = f"{r['latency_ms']:.4f} ± {r['stddev_ms']:.4f}"
        tput_str = f"{r['throughput']:.0f} tok/s"
        print(f"{kernel_name:<30} {r['num_tokens']:<10} {lat_str:<20} {tput_str:<15}")
    print("─" * 90)

EOF "$RESULTS_FILE"
}

# List generated files
list_files() {
    print_header "GENERATED FILES"

    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        ls -lh "$OUTPUT_DIR"/ | tail -n +2 | awk '{printf "  %-40s %8s\n", $9, $5}'
        echo ""
    fi
}

# Show profiling instructions
show_profiling_tips() {
    if [ "$PROFILE" = "true" ]; then
        print_header "PROFILING INSTRUCTIONS"

        echo ""
        echo "View trace files with Nsight Systems GUI:"
        echo "  nsys-ui $OUTPUT_DIR/moe_triton_*.nsys-rep"
        echo ""
        echo "Generate statistical reports:"
        echo "  nsys stats $OUTPUT_DIR/moe_triton_*.nsys-rep"
        echo ""
        echo "Export to SQLite for analysis:"
        echo "  nsys export -t sqlite $OUTPUT_DIR/moe_triton_*.nsys-rep"
        echo ""
    fi
}

# Main
main() {
    print_header "MoE TRITON KERNEL BENCHMARK"

    echo ""
    echo "Configuration:"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Profiling: $PROFILE"
    echo "  Repeats: $NUM_REPEATS"
    echo "  Warmup iterations: $WARMUP"
    echo ""

    check_requirements
    setup_output
    run_benchmark

    if [ $? -eq 0 ]; then
        analyze_results
        list_files
        show_profiling_tips

        print_header "BENCHMARK COMPLETE ✓"
        echo ""
        echo "Next steps:"
        echo "  1. Check results: cat $OUTPUT_DIR/benchmark_results.json"
        echo "  2. View profiles: nsys-ui $OUTPUT_DIR/*.nsys-rep (if profiling enabled)"
        echo "  3. Compare runs: diff <(jq . $OUTPUT_DIR/benchmark_results.json) ..."
        echo ""
    else
        print_error "Benchmark failed"
        exit 1
    fi
}

# Run
main
