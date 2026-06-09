"""PyTorch CPU conv benchmark — counterpart to bench_nnlib.jl (issue #234).

Run with 4 threads:
    uv run --project . python bench_torch.py
"""
import time

import torch

NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
torch.set_grad_enabled(False)

# (name, in_ch, out_ch, kernel, stride, pad, H, W, batch)
CASES = [
    ("issue234 7x7 s2 p3", 3, 64, 7, 2, 3, 224, 224, 2),
    ("3x3 s1 p1 c64", 64, 64, 3, 1, 1, 56, 56, 2),
    ("3x3 s1 p1 c128", 128, 128, 3, 1, 1, 28, 28, 2),
    ("1x1 s1 p0 c256", 256, 256, 1, 1, 0, 14, 14, 2),
]


def bench(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]  # median, in seconds


def main():
    print("torch version:", torch.__version__)
    print("num threads:  ", torch.get_num_threads())
    print(f"{'case':<22}{'fwd (ms)':>12}")
    for name, ci, co, k, s, p, H, W, b in CASES:
        conv = torch.nn.Conv2d(ci, co, k, stride=s, padding=p, bias=True)
        x = torch.randn(b, ci, H, W)
        fwd = bench(lambda: conv(x))
        print(f"{name:<22}{fwd * 1e3:>12.4f}")


if __name__ == "__main__":
    main()
