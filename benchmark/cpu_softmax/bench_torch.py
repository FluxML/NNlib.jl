"""PyTorch counterpart of bench_softmax.jl: `torch.softmax` on the CPU.

Reports the wall time in ms, on one thread and on `--threads`: the minimum over the
samples that fit in the time budget. PyTorch allocates its output on every call.

    uv run --project . python bench_torch.py --threads 16
    uv run --project . python bench_torch.py --dtypes f64 --sizes classifier
"""
import argparse
import time

import torch

# (name, size, dims), as in bench_softmax.jl
CONFIGS = [
    ("attention", (256, 256, 256), 1),
    ("long slices", (1024, 1024, 16), 1),
    ("classifier", (1000, 512), 1),
    ("few classes", (10, 65536), 1),
    ("rows", (2048, 512), 2),
    ("segmentation", (128, 128, 21, 16), 3),
    ("2^16 elements", (64, 1024), 1),
    ("2^10 elements", (16, 64), 1),
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
    p.add_argument("--sizes", default="all")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = p.parse_args()

    configs = CONFIGS if args.sizes == "all" else [c for c in CONFIGS if c[0] in args.sizes.split(",")]
    print("PyTorch:", torch.__version__, " threads:", args.threads)
    for dtype_name in args.dtypes.split(","):
        print(f"\n### dtype={dtype_name}  (ms)")
        print(f"{'config':<15}{'size':<17}{'dims':<5}|{'1 thread':>12}{f'{args.threads} threads':>12}")
        for name, size, dims in configs:
            # Julia arrays are column-major: the same memory layout has the dimensions reversed
            x = torch.randn(*reversed(size), dtype=DTYPES[dtype_name])
            dim = len(size) - dims
            times = []
            for threads in (1, args.threads):
                torch.set_num_threads(threads)
                with torch.no_grad():
                    times.append(timed(lambda: torch.softmax(x, dim), args.seconds))
            print(f"{name:<15}{'x'.join(map(str, size)):<17}{dims:<5}|{times[0]:>12.3f}{times[1]:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
