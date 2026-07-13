"""
Modal profiling script for the DeepSeek-V3 MoE baseline (flashinfer_baseline.py).
Uses NVIDIA Nsight Systems (nsys) for GPU profiling with NVTX annotations.

Usage:
    modal run modal_profile.py                          # default: 64 tokens
    modal run modal_profile.py --tokens 128
    modal run modal_profile.py --tokens 128 --warmup 5 --iters 20 --output my_trace.nsys-rep

Output: a .nsys-rep file you can open locally with Nsight Systems UI.
    nsys-ui moe_profile.nsys-rep
"""

import modal

app = modal.App("flashinfer-moe-nsight")

# NVIDIA PyTorch container ships with nsys pre-installed.
image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.08-py3")

# ------------------------------------------------------------------ #
# Inner script that runs *inside* the nsys profile subprocess.
# Written as a string so we can template tokens/warmup/iters into it.
# ------------------------------------------------------------------ #
_INNER_SCRIPT = r'''
import torch
import torch.cuda.nvtx as nvtx

T       = {tokens}
WARMUP  = {warmup}
ITERS   = {iters}
H       = 7168
I_DIM   = 2048
E_GLOB  = 256
E_LOC   = 32
BLOCK   = 128

device = "cuda"

# ---- tensor construction ------------------------------------------------ #

def make_inputs():
    nhb  = H // BLOCK
    ng1b = (2 * I_DIM) // BLOCK
    nib  = I_DIM // BLOCK

    routing_logits = torch.randn(T, E_GLOB, dtype=torch.float16, device=device)
    routing_bias   = torch.randn(E_GLOB,    dtype=torch.float16, device=device)

    hidden_states       = torch.randn(T, H, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.ones(nhb, T, dtype=torch.float32, device=device)

    gemm1_weights       = torch.randn(E_LOC, 2 * I_DIM, H, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.ones(E_LOC, ng1b, nhb, dtype=torch.float32, device=device)

    gemm2_weights       = torch.randn(E_LOC, H, I_DIM, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.ones(E_LOC, nhb, nib,  dtype=torch.float32, device=device)

    return (routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            0, 2.5)

# ---- baseline run ------------------------------------------------------- #

@torch.no_grad()
def run(routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):

    E_global  = routing_logits.shape[1]
    T_local   = routing_logits.shape[0]
    E_local   = gemm1_weights.shape[0]
    TOP_K     = 8
    N_GROUP   = 8
    TOPK_GRP  = 4

    # -- FP8 dequant activations ------------------------------------------
    with nvtx.range("dequant_activations"):
        A_fp32 = hidden_states.to(torch.float32)
        A_scl  = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()
        A_scl  = A_scl.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T_local, H).contiguous()
        A      = A_fp32 * A_scl

    # -- FP8 dequant W13 --------------------------------------------------
    with nvtx.range("dequant_W13"):
        W13 = gemm1_weights.to(torch.float32)
        S13 = gemm1_weights_scale.to(torch.float32)
        S13 = torch.repeat_interleave(S13, BLOCK, dim=1)
        S13 = torch.repeat_interleave(S13, BLOCK, dim=2)
        W13 = W13 * S13

    # -- FP8 dequant W2 ---------------------------------------------------
    with nvtx.range("dequant_W2"):
        W2 = gemm2_weights.to(torch.float32)
        S2 = gemm2_weights_scale.to(torch.float32)
        S2 = torch.repeat_interleave(S2, BLOCK, dim=1)
        S2 = torch.repeat_interleave(S2, BLOCK, dim=2)
        W2 = W2 * S2

    # -- Routing ----------------------------------------------------------
    with nvtx.range("routing"):
        logits      = routing_logits.to(torch.float32)
        bias        = routing_bias.to(torch.float32).reshape(-1)
        s           = 1.0 / (1.0 + torch.exp(-logits))
        s_wb        = s + bias
        grp_sz      = E_global // N_GROUP
        s_grp       = s_wb.view(T_local, N_GROUP, grp_sz)
        top2, _     = torch.topk(s_grp, k=2, dim=2, largest=True, sorted=False)
        g_scores    = top2.sum(dim=2)
        _, g_idx    = torch.topk(g_scores, k=TOPK_GRP, dim=1, largest=True, sorted=False)
        g_mask      = torch.zeros_like(g_scores)
        g_mask.scatter_(1, g_idx, 1.0)
        score_mask  = g_mask.unsqueeze(2).expand(T_local, N_GROUP, grp_sz).reshape(T_local, E_global)
        neg_inf     = torch.finfo(torch.float32).min
        pruned      = s_wb.masked_fill(score_mask == 0, neg_inf)
        _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)
        M           = torch.zeros_like(s)
        M.scatter_(1, topk_idx, 1.0)
        weights     = s * M
        weights     = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor

    # -- Expert compute ---------------------------------------------------
    output     = torch.zeros((T_local, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        with nvtx.range(f"expert_{ge}"):
            sel  = (topk_idx == ge).any(dim=1)
            if not sel.any():
                continue
            tidx  = torch.nonzero(sel, as_tuple=False).squeeze(1)
            A_e   = A.index_select(0, tidx)
            G1    = A_e.matmul(W13[le].t())
            X1, X2 = G1[:, :I_DIM], G1[:, I_DIM:]
            C     = (X2 / (1.0 + torch.exp(-X2))) * X1
            O     = C.matmul(W2[le].t())
            w_tok = weights.index_select(0, tidx)[:, ge]
            output.index_add_(0, tidx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)

# ---- main --------------------------------------------------------------- #

args = make_inputs()

print(f"Warming up ({WARMUP} iters) ...")
for _ in range(WARMUP):
    run(*args)
torch.cuda.synchronize()

print(f"Profiling ({ITERS} iters) ...")
for i in range(ITERS):
    with nvtx.range(f"iter_{i}"):
        run(*args)
torch.cuda.synchronize()

print("Done.")
'''


# ------------------------------------------------------------------ #
# Modal function
# ------------------------------------------------------------------ #

@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def profile_nsight(tokens: int = 64, warmup: int = 3, iters: int = 10) -> bytes:
    import os
    import subprocess
    import tempfile

    # Write the inner script to a temp file with the requested parameters.
    script = _INNER_SCRIPT.format(tokens=tokens, warmup=warmup, iters=iters)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(script)
        inner_path = f.name

    output_base = "/tmp/moe_profile"

    cmd = [
        "nsys", "profile",
        "--output",            output_base,
        "--force-overwrite",   "true",
        "--trace",             "cuda,nvtx,cudnn,cublas",
        "--cuda-memory-usage", "true",
        "--gpu-metrics-device","all",
        "python", inner_path,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)

    rep_file = output_base + ".nsys-rep"
    if not os.path.exists(rep_file):
        raise RuntimeError(
            f"nsys did not produce {rep_file}. "
            f"Return code: {result.returncode}"
        )

    with open(rep_file, "rb") as f:
        return f.read()


# ------------------------------------------------------------------ #
# Local entry point – saves the .nsys-rep locally
# ------------------------------------------------------------------ #

@app.local_entrypoint()
def main(
    tokens: int = 64,
    warmup: int = 3,
    iters: int = 10,
    output: str = "moe_profile.nsys-rep",
):
    print(f"Launching profiling on Modal H100 (T={tokens}, warmup={warmup}, iters={iters}) ...")
    data = profile_nsight.remote(tokens=tokens, warmup=warmup, iters=iters)

    with open(output, "wb") as f:
        f.write(data)

    size_mb = len(data) / 1024 / 1024
    print(f"\nTrace saved to '{output}'  ({size_mb:.1f} MB)")
    print("Open locally with:")
    print(f"  nsys-ui {output}")
