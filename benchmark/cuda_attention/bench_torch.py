"""PyTorch counterpart of bench_attention.jl: `scaled_dot_product_attention` on the GPU,
with each of its kernels.

    math        materialized: matmul, softmax, matmul. Memory is O(seq²).
    efficient   the memory-efficient kernel (CUTLASS), which NNlib's fused kernel follows.
    flash       FlashAttention.
    cudnn       cuDNN's fused kernel, the one NNlib hands half precision to.

Reports the forward and the forward+backward wall time in ms, GPU-synchronized: the
minimum over the samples that fit in the time budget. "n/s" = the kernel does not take
that input.

    uv run --project . python bench_torch.py
    uv run --project . python bench_torch.py --dtypes f32 --causal false --sizes gpt2-ish
"""
import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# (name, head_dim E, seq_len L, nheads H, batch B), as in bench_attention.jl
CONFIGS = [
    ("tiny", 64, 128, 4, 8),
    ("small", 64, 512, 8, 8),
    ("gpt2-ish", 64, 1024, 12, 4),
    ("llama3 L=2k", 128, 2048, 32, 1),
    ("llama3 L=4k", 128, 4096, 32, 1),
    ("llama3 L=8k", 128, 8192, 32, 1),
]
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def timed(fn, seconds):
    """Minimum time in ms, or None if `fn` does not run."""
    try:
        fn()  # warmup
        torch.cuda.synchronize()
        best, start = float("inf"), time.perf_counter()
        while time.perf_counter() - start < seconds:
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        return best * 1e3
    except RuntimeError:
        return None
    finally:
        torch.cuda.empty_cache()


def fmt(x):
    return "n/s" if x is None else f"{x:.3f}"


def run(dtype_name, causal, backends, configs, seconds):
    dtype = DTYPES[dtype_name]
    print(f"\n{'#' * 100}\n### dtype={dtype_name}  causal={causal}\n{'#' * 100}")
    rows = []
    for name, E, L, H, B in configs:
        # PyTorch's layout is (batch, heads, seq, head_dim)
        q, k, v = (torch.randn(B, H, L, E, device="cuda", dtype=dtype) for _ in range(3))
        qg, kg, vg = (x.clone().requires_grad_() for x in (q, k, v))
        row = {}
        for b in backends:
            def fwd():
                with torch.no_grad(), sdpa_kernel(BACKENDS[b]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

            def grad():
                with sdpa_kernel(BACKENDS[b]):
                    y = F.scaled_dot_product_attention(qg, kg, vg, is_causal=causal)
                return torch.autograd.grad(y.sum(), (qg, kg, vg))

            row[b] = (timed(fwd, seconds), timed(grad, seconds))
        rows.append((name, L, H, row))
        del q, k, v, qg, kg, vg
        torch.cuda.empty_cache()
    for title, idx in (("FORWARD (ms)", 0), ("FULL GRADIENT, fwd+bwd (ms)", 1)):
        print(f"\n-- {title} --")
        print(f"{'config':<13}{'L':>6}{'H':>4} |" + "".join(f"{b:>12}" for b in backends))
        for name, L, H, row in rows:
            print(f"{name:<13}{L:>6}{H:>4} |" + "".join(f"{fmt(row[b][idx]):>12}" for b in backends), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtypes", default="f16,bf16,f32")
    p.add_argument("--backends", default=",".join(BACKENDS))
    p.add_argument("--causal", default="false,true")
    p.add_argument("--sizes", default="all")
    p.add_argument("--seconds", type=float, default=1.0)
    args = p.parse_args()

    configs = CONFIGS if args.sizes == "all" else [c for c in CONFIGS if c[0] in args.sizes.split(",")]
    print("GPU:    ", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__, " cuDNN:", torch.backends.cudnn.version())
    for dtype_name in args.dtypes.split(","):
        for causal in args.causal.split(","):
            run(dtype_name, causal == "true", args.backends.split(","), configs, args.seconds)


if __name__ == "__main__":
    main()
