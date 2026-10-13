"""PyTorch counterpart of bench_attention.jl: `scaled_dot_product_attention` on the CPU,
where PyTorch materializes the attention weights (matmul, softmax, matmul).

Reports the forward wall time in ms: the minimum over the samples that fit in the time
budget.

    uv run --project . python bench_torch.py --threads 16
    uv run --project . python bench_torch.py --causal false --sizes gpt2-ish
"""
import argparse
import time

import torch
import torch.nn.functional as F

# (name, head_dim E, seq_len L, nheads H, batch B), as in bench_attention.jl
CONFIGS = [
    ("tiny", 64, 128, 4, 8),
    ("small", 64, 512, 8, 8),
    ("gpt2-ish", 64, 1024, 12, 4),
    ("llama3 L=2k", 128, 2048, 32, 1),
    ("llama3 L=4k", 128, 4096, 32, 1),
    ("llama3 L=8k", 128, 8192, 32, 1),
]
DTYPES = {"f32": torch.float32, "f64": torch.float64}


def timed(fn, seconds):
    fn()  # warmup
    best, start = float("inf"), time.perf_counter()
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtypes", default="f32")
    p.add_argument("--causal", default="false,true")
    p.add_argument("--sizes", default="all")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    configs = CONFIGS if args.sizes == "all" else [c for c in CONFIGS if c[0] in args.sizes.split(",")]
    print("PyTorch:", torch.__version__, " threads:", torch.get_num_threads())
    for dtype_name in args.dtypes.split(","):
        for causal in args.causal.split(","):
            is_causal = causal == "true"
            print(f"\n### dtype={dtype_name}  causal={is_causal}")
            print(f"{'config':<13}{'L':>6}{'H':>4} |{'torch ms':>12}")
            for name, E, L, H, B in configs:
                # PyTorch's layout is (batch, heads, seq, head_dim)
                q, k, v = (torch.randn(B, H, L, E, dtype=DTYPES[dtype_name]) for _ in range(3))
                t = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=is_causal), args.seconds)
                print(f"{name:<13}{L:>6}{H:>4} |{t:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
