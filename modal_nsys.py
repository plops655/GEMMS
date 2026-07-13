"""
Modal Nsight Systems profiling script for persistent_kernel2.cu (v2)

Compiles the CUDA kernel on a remote GPU, runs it under nsys to capture
a system-level timeline (kernel durations, memory copies, gaps between kernels),
and downloads the .nsys-rep file locally.

Usage:
    modal run modal_nsys.py                          # default: 64 tokens, 10 iters
    modal run modal_nsys.py --tokens 128 --iters 20
    modal run modal_nsys.py --output my_trace.nsys-rep

Open locally with:
    nsys-ui moe_nsys_profile_v2.nsys-rep
"""

import modal
from pathlib import Path

app = modal.App("persistent-moe-nsys")

# CUDA 12.x container with nvcc and nsys
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3")
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/src",
        copy=True,
        ignore=lambda p: not (str(p).endswith(".cu") or str(p).endswith(".h")),
    )
)

# ------------------------------------------------------------------ #
# Test harness — compiled alongside the kernel
# ------------------------------------------------------------------ #

_TEST_HARNESS = r'''
#include "common.h"
#include "moe_constants.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Forward-declare the launch function from persistent_kernel2.cu
extern void launch_persistent_moe_v2(
    const float*           routing_logits,
    const float*           routing_bias,
    const __nv_fp8_e4m3*   hidden_states,
    const float*           hidden_states_scale,
    const __nv_fp8_e4m3*   gemm1_weights,
    const float*           gemm1_weights_scale,
    const __nv_fp8_e4m3*   gemm2_weights,
    const float*           gemm2_weights_scale,
    int                    T,
    int                    local_expert_offset,
    float                  routed_scaling_factor,
    float*                 output,
    cudaStream_t           stream);

// Fill a device buffer with random-ish FP8 values (small magnitude)
void fill_fp8(void* d_ptr, size_t n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        h[i] = 0.01f * ((float)(rand() % 201 - 100));

    __nv_fp8_e4m3* h_fp8 = (__nv_fp8_e4m3*)malloc(n * sizeof(__nv_fp8_e4m3));
    for (size_t i = 0; i < n; i++)
        h_fp8[i] = __nv_fp8_e4m3(h[i]);

    cudaMemcpy(d_ptr, h_fp8, n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    free(h);
    free(h_fp8);
}

void fill_float(void* d_ptr, size_t n, float val) {
    float* h = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        h[i] = val;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

void fill_float_rand(void* d_ptr, size_t n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        h[i] = 0.01f * ((float)(rand() % 201 - 100));
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

int main(int argc, char** argv) {
    int T = 64;
    int WARMUP = 3;
    int ITERS = 10;
    if (argc > 1) T = atoi(argv[1]);
    if (argc > 2) WARMUP = atoi(argv[2]);
    if (argc > 3) ITERS = atoi(argv[3]);

    printf("persistent_kernel2 nsys harness: T=%d, warmup=%d, iters=%d\n", T, WARMUP, ITERS);
    printf("  H_DIM=%d, I_DIM=%d, E_LOCAL=%d, NUM_SMS=%d\n", H_DIM, I_DIM, E_LOCAL, NUM_SMS);

    srand(42);

    // Allocate inputs
    float *d_routing_logits, *d_routing_bias;
    __nv_fp8_e4m3 *d_hidden, *d_w1, *d_w2;
    float *d_hidden_scale, *d_w1_scale, *d_w2_scale;
    float *d_output;

    CUDA_CHECK(cudaMalloc(&d_routing_logits, (size_t)T * E_GLOBAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_routing_bias,   (size_t)E_GLOBAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden,         (size_t)T * H_DIM * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_hidden_scale,   (size_t)NUM_H_BLOCKS * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1,             (size_t)E_LOCAL * 2 * I_DIM * H_DIM * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_w1_scale,       (size_t)E_LOCAL * NUM_2I_BLOCKS * NUM_H_BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2,             (size_t)E_LOCAL * H_DIM * I_DIM * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_w2_scale,       (size_t)E_LOCAL * NUM_H_BLOCKS * NUM_I_BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,         (size_t)T * H_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0,       (size_t)T * H_DIM * sizeof(float)));

    // Fill with test data
    fill_float_rand(d_routing_logits, (size_t)T * E_GLOBAL);
    fill_float_rand(d_routing_bias,   (size_t)E_GLOBAL);
    fill_fp8(d_hidden,                (size_t)T * H_DIM);
    fill_float(d_hidden_scale,        (size_t)NUM_H_BLOCKS * T, 1.0f);
    fill_fp8(d_w1,                    (size_t)E_LOCAL * 2 * I_DIM * H_DIM);
    fill_float(d_w1_scale,            (size_t)E_LOCAL * NUM_2I_BLOCKS * NUM_H_BLOCKS, 1.0f);
    fill_fp8(d_w2,                    (size_t)E_LOCAL * H_DIM * I_DIM);
    fill_float(d_w2_scale,            (size_t)E_LOCAL * NUM_H_BLOCKS * NUM_I_BLOCKS, 1.0f);

    printf("  Allocated %.1f MB of GPU memory\n",
        ((double)T*E_GLOBAL*4 + E_GLOBAL*4 + T*H_DIM + NUM_H_BLOCKS*T*4 +
         (double)E_LOCAL*2*I_DIM*H_DIM + (double)E_LOCAL*NUM_2I_BLOCKS*NUM_H_BLOCKS*4 +
         (double)E_LOCAL*H_DIM*I_DIM + (double)E_LOCAL*NUM_H_BLOCKS*NUM_I_BLOCKS*4 +
         T*H_DIM*4) / (1024.0*1024.0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    printf("  Warmup (%d iters)...\n", WARMUP);
    for (int i = 0; i < WARMUP; i++) {
        launch_persistent_moe_v2(
            d_routing_logits, d_routing_bias,
            d_hidden, d_hidden_scale,
            d_w1, d_w1_scale,
            d_w2, d_w2_scale,
            T, 0, 2.5f,
            d_output, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Profiled iterations
    printf("  Profiled run (%d iters)...\n", ITERS);
    for (int i = 0; i < ITERS; i++) {
        launch_persistent_moe_v2(
            d_routing_logits, d_routing_bias,
            d_hidden, d_hidden_scale,
            d_w1, d_w1_scale,
            d_w2, d_w2_scale,
            T, 0, 2.5f,
            d_output, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("  Done.\n");

    // Cleanup
    cudaFree(d_routing_logits);
    cudaFree(d_routing_bias);
    cudaFree(d_hidden);
    cudaFree(d_hidden_scale);
    cudaFree(d_w1);
    cudaFree(d_w1_scale);
    cudaFree(d_w2);
    cudaFree(d_w2_scale);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    return 0;
}
'''


# ------------------------------------------------------------------ #
# Modal function — compile + profile on remote GPU
# ------------------------------------------------------------------ #

@app.function(
    image=image,
    gpu="B200",
    timeout=900,
)
def profile_nsys(tokens: int = 64, warmup: int = 3, iters: int = 10) -> bytes:
    import os
    import subprocess

    os.chdir("/src")

    # Write the test harness
    with open("test_harness.cu", "w") as f:
        f.write(_TEST_HARNESS)

    # Detect GPU arch
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    arch = result.stdout.strip().replace(".", "")
    print(f"Detected GPU compute capability: sm_{arch}")

    # Compile: persistent_kernel2.cu + test_harness.cu
    compile_cmd = [
        "nvcc",
        "-o", "moe_test",
        "persistent_kernel2.cu", "test_harness.cu",
        f"-arch=sm_{arch}",
        "-std=c++17",
        "-O2",
        "--expt-relaxed-constexpr",
        "-lineinfo",
    ]
    print("Compiling:", " ".join(compile_cmd))
    r = subprocess.run(compile_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        raise RuntimeError(f"nvcc failed with code {r.returncode}")
    print("Compilation successful.")

    # Run under nsys
    nsys_output = "/tmp/moe_nsys_profile_v2"
    nsys_cmd = [
        "nsys", "profile",
        "--output", nsys_output,
        "--force-overwrite", "true",
        "--trace", "cuda,osrt",          # CUDA API + OS runtime
        "--cuda-memory-usage", "true",
        "--gpu-metrics-device", "all",   # SM utilization, memory BW, etc.
        "--stats", "true",               # print summary stats at end
        "./moe_test", str(tokens), str(warmup), str(iters),
    ]
    print("Profiling:", " ".join(nsys_cmd))
    r = subprocess.run(nsys_cmd, capture_output=True, text=True)
    print("NSYS STDOUT:", r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
    if r.stderr:
        print("NSYS STDERR:", r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr)

    rep_file = nsys_output + ".nsys-rep"
    if not os.path.exists(rep_file):
        # nsys sometimes adds a numeric suffix
        import glob
        candidates = sorted(glob.glob(nsys_output + "*.nsys-rep"))
        if candidates:
            rep_file = candidates[-1]
    if not os.path.exists(rep_file):
        raise RuntimeError(
            f"nsys did not produce {rep_file}. Return code: {r.returncode}"
        )

    print(f"Profile size: {os.path.getsize(rep_file) / 1024 / 1024:.1f} MB")
    with open(rep_file, "rb") as f:
        return f.read()


# ------------------------------------------------------------------ #
# Local entry point
# ------------------------------------------------------------------ #

@app.local_entrypoint()
def main(
    tokens: int = 64,
    warmup: int = 3,
    iters: int = 10,
    output: str = "moe_nsys_profile_v2.nsys-rep",
):
    print(f"Launching nsys profiling on Modal B200 (T={tokens}, warmup={warmup}, iters={iters}) ...")
    data = profile_nsys.remote(tokens=tokens, warmup=warmup, iters=iters)

    with open(output, "wb") as f:
        f.write(data)

    size_mb = len(data) / 1024 / 1024
    print(f"\nTrace saved to '{output}'  ({size_mb:.1f} MB)")
    print("Open locally with:")
    print(f"  nsys-ui {output}")
