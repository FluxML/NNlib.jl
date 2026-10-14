# CPU softmax benchmark — Julia tasks, KernelAbstractions kernels and PyTorch

Compares implementations of `softmax!` for CPU arrays, to tell how NNlib should thread
the operators that still run on one core
([#790](https://github.com/FluxML/NNlib.jl/issues/790)): with Julia tasks, or with a
KernelAbstractions kernel on the CPU backend.

| Name | Function | Approach |
|---|---|---|
| `broadcast` | `NNlib._softmax!` | the generic implementation: three broadcasts on one thread, with the `exp` of Base. It is what `softmax!` ran on the CPU up to NNlib 0.9.47 |
| `serial` | `NNlib.softmax!` under `NNlib.@disallow_spawns` | loops written for the CPU, which the compiler vectorizes, on one task |
| `spawn` | `NNlib.softmax!` | the same loops on Julia tasks |
| `ka` | `softmax_ka!`, in `ka_softmax.jl` | the same loops as a kernel on the CPU backend: a work-item for each chunk of slices, workgroup size 1 |
| `ka-slice` | `softmax_ka_slice!` | a kernel with a work-item for each slice, the workgroup size left to the backend. Contiguous slices only |
| `ka-slice-1` | | the same with a workgroup size of 1 |
| `ka-unsafe` | | `ka-slice` declared with `unsafe_indices=true` |
| `ka-unsafe-1n` | | `ka-unsafe` with loops over `1:n` in every work-item, instead of over the indices of its slice: the case of [KernelAbstractions.jl#890](https://github.com/JuliaGPU/KernelAbstractions.jl/issues/890) |
| PyTorch | `torch.softmax`, in `bench_torch.py` | |

The loops of `spawn` and of `ka` do the same work. An array is seen as
`(pre, n, post)`, with the dimensions of the softmax in the middle: for `pre == 1` a
slice is contiguous and the loops run along it, otherwise they run along the first
dimension over a tile of rows. What differs is who divides the work among the threads.

The CPU backend of KernelAbstractions is two different things. In 0.9 a kernel is
compiled by Julia and run by one task for each thread, each with a fixed share of the
workgroups. In 0.10 it is compiled to SPIR-V and handed to [PoCL](https://portablecl.org),
an OpenCL implementation with a compiler and threads of its own. `ka_softmax.jl` is
written for the second, see [below](#what-the-kernel-needs-on-pocl), and runs on both.

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=. bench_softmax.jl
julia --threads=1 --project=. bench_loops.jl        # the loops one at a time, Julia against the kernel

uv sync                                             # PyTorch, from pyproject.toml
uv run --project . python bench_torch.py --threads 16
```

The Julia environment uses the NNlib of this checkout (`[sources]` in `Project.toml`),
whose `[compat]` holds KernelAbstractions at 0.9. For 0.10 the benchmark takes a copy of
the checkout with `KernelAbstractions = "0.9.34, 0.10"` in the `[compat]` of NNlib.

`bench_softmax.jl` and `bench_torch.py` take flags, so that a single combination can be
measured:

```bash
julia --threads=16 --project=. bench_softmax.jl --dtypes f64 --impls spawn,ka --sizes classifier
julia --threads=16 --project=. bench_softmax.jl --inplace true
julia --project=. bench_softmax.jl --help     # all options
julia --project=. bench_softmax.jl --list     # available configs / dtypes / impls
```

| flag | values | default |
|---|---|---|
| `--dtypes` | `f32`,`f64` | `f32` |
| `--impls` | the names above but PyTorch | all |
| `--sizes` | config names (see `--list`) | all |
| `--inplace` | `false`,`true`: whether the output is the input | `false` |
| `--seconds` | per-measurement time budget | `2.0` |

Each cell is the minimum over the samples that fit in the time budget, one evaluation
per sample, in ms. The output array is allocated beforehand, except by PyTorch, which
allocates one in every call. Before timing, every implementation is checked against
`broadcast`, out of place and in place.

Configs (`softmax` along `dims`):

```
attention       256x256x256     dims=1   weights of 256 queries on 256 keys
long slices     1024x1024x16    dims=1
classifier      1000x512        dims=1   1000 classes, batch of 512
few classes     10x65536        dims=1
rows            2048x512        dims=2   not contiguous
segmentation    128x128x21x16   dims=3   not contiguous: 21 classes for each pixel
2^16 elements   64x1024         dims=1   below the 2^17 elements from which NNlib spawns tasks
2^10 elements   16x64           dims=1
```

## Results — Threadripper PRO 9955WX (16 cores, AVX-512), Julia 1.13.1, 16 threads

Raw output: [`results_tr9955wx_ka09.txt`](results_tr9955wx_ka09.txt) (KernelAbstractions
0.9.45), [`results_tr9955wx_ka010.txt`](results_tr9955wx_ka010.txt) (0.10.0, PoCL 7.2),
[`results_tr9955wx_loops_ka010.txt`](results_tr9955wx_loops_ka010.txt) and
[`results_tr9955wx_torch213.txt`](results_tr9955wx_torch213.txt) (PyTorch 2.13.0).

`Float32`, times in ms, lower is better. The first three columns are from the run with
0.9.45; they are the same code in both.

Out of place:

| config | broadcast | serial | spawn | ka, 0.9.45 | ka, 0.10.0 | PyTorch, 1 thread | PyTorch, 16 threads |
|---|--:|--:|--:|--:|--:|--:|--:|
| attention | 57.9 | 3.81 | 1.63 | 1.91 | 1.76 | 17.8 | 4.05 |
| long slices | 57.8 | 3.75 | 1.65 | 1.97 | 1.80 | 17.2 | 4.23 |
| classifier | 1.49 | 0.099 | 0.019 | 0.019 | 0.029 | 0.167 | 0.015 |
| few classes | 2.03 | 0.762 | 0.072 | 0.079 | 0.108 | 6.32 | 0.430 |
| rows | 3.02 | 0.236 | 0.042 | 0.040 | 0.092 | 0.930 | 0.076 |
| segmentation | 22.8 | 2.00 | 0.109 | 0.158 | 0.279 | 3.20 | 0.486 |
| 2^16 elements | 0.198 | 0.018 | 0.018 | 0.021 | 0.018 | 0.027 | 0.005 |
| 2^10 elements | 0.0035 | 0.0008 | 0.0007 | 0.0008 | 0.0008 | 0.001 | 0.003 |

In place:

| config | broadcast | serial | spawn | ka, 0.9.45 | ka, 0.10.0 |
|---|--:|--:|--:|--:|--:|
| attention | 57.9 | 3.05 | 0.27 | 0.41 | 0.53 |
| long slices | 57.0 | 4.18 | 0.28 | 0.74 | 0.55 |
| classifier | 1.49 | 0.097 | 0.019 | 0.021 | 0.030 |
| few classes | 2.03 | 0.703 | 0.070 | 0.067 | 0.107 |
| rows | 3.00 | 0.222 | 0.039 | 0.042 | 0.087 |
| segmentation | 22.2 | 1.22 | 0.107 | 0.148 | 0.264 |

`Float64`, out of place:

| config | broadcast | serial | spawn | ka, 0.9.45 | ka, 0.10.0 | PyTorch, 1 thread | PyTorch, 16 threads |
|---|--:|--:|--:|--:|--:|--:|--:|
| attention | 71.4 | 8.41 | 4.53 | 4.82 | 4.63 | 36.5 | 7.58 |
| classifier | 1.68 | 0.217 | 0.030 | 0.029 | 0.046 | 0.399 | 0.030 |
| rows | 3.67 | 0.629 | 0.093 | 0.070 | 0.174 | 2.05 | 0.165 |
| segmentation | 26.8 | 4.75 | 0.687 | 0.968 | 0.832 | 15.0 | 3.12 |

The time of the first call, compilation included, on a 256×4096 `Float32` matrix: 0.6 s
for `spawn`, 0.7 s for `ka` with 0.9.45 and 4.0 s with 0.10.0. PoCL compiles a kernel
for each element type, and one more for the kernel that works in place.

### The kernels with a work-item for each slice

`Float32`, out of place, with KernelAbstractions 0.10.0:

| config | ka | ka-slice | ka-slice-1 | ka-unsafe | ka-unsafe-1n |
|---|--:|--:|--:|--:|--:|
| attention | 1.76 | 1.69 | 1.99 | 1.69 | **6.10** |
| long slices | 1.80 | 1.63 | 1.82 | 1.62 | **4.60** |
| classifier | 0.029 | 0.028 | 0.035 | 0.029 | **0.042** |
| few classes | 0.108 | 0.114 | 0.146 | 0.114 | 0.116 |
| 2^10 elements | 0.0008 | 0.0038 | 0.0076 | 0.0038 | 0.0038 |

and with 0.9.45:

| config | ka | ka-slice | ka-slice-1 | ka-unsafe | ka-unsafe-1n |
|---|--:|--:|--:|--:|--:|
| attention | 1.91 | 1.88 | 1.85 | 1.79 | 1.95 |
| long slices | 1.97 | 1.95 | 1.99 | 1.81 | 2.05 |
| classifier | 0.019 | **0.105** | 0.021 | **0.104** | **0.103** |
| few classes | 0.079 | 0.079 | 0.086 | 0.078 | 0.073 |
| 2^10 elements | 0.0008 | 0.0009 | **0.0114** | 0.0009 | 0.0009 |

In place these kernels take the array as two arguments, and with 0.10.0 `attention` goes
from 1.69 to 3.52 ms where `ka` goes from 1.76 to 0.53.

### The loops one at a time

`bench_loops.jl`, on one thread: nanoseconds per element of `Float32` slices of 256
elements, for a loop compiled by Julia and for the same loop in a kernel compiled by
PoCL (KernelAbstractions 0.10.0). With 0.9.45 the two columns are the same code.

| loop | Julia | PoCL | PoCL ÷ Julia |
|---|--:|--:|--:|
| `exp`, with `round` | 0.141 | 0.404 | 2.9× |
| `exp`, adding and subtracting a constant | 0.154 | 0.426 | 2.8× |
| `max`, `@fastmath` | 0.028 | 0.995 | 35× |
| `max`, `@fastmath`, entries clamped (NNlib) | 0.030 | 1.005 | 34× |
| `max`, on the bit patterns | 0.028 | 0.033 | 1.2× |
| sum | 0.028 | 0.030 | 1.1× |
| scaling | 0.030 | 0.033 | 1.1× |
| softmax of a slice, as in `ka_softmax.jl` | 0.211 | 0.457 | 2.2× |
| the same in place, array given once | 0.207 | 0.465 | 2.3× |
| the same in place, array given twice | 0.207 | 3.501 | 17× |

## Results — Apple M1 Pro (6 + 2 cores), Julia 1.13.1, 8 threads

Raw output: [`results_m1pro_ka09.txt`](results_m1pro_ka09.txt),
[`results_m1pro_ka010.txt`](results_m1pro_ka010.txt),
[`results_m1pro_loops_ka010.txt`](results_m1pro_loops_ka010.txt) and
[`results_m1pro_torch214.txt`](results_m1pro_torch214.txt) (PyTorch 2.14.1).

`Float32`, out of place, times in ms:

| config | broadcast | serial | spawn | ka, 0.9.45 | ka, 0.10.0 | PyTorch, 1 thread | PyTorch, 8 threads |
|---|--:|--:|--:|--:|--:|--:|--:|
| attention | 75.5 | 12.6 | 2.12 | 2.84 | 3.05 | 24.6 | 4.60 |
| long slices | 75.7 | 12.0 | 2.00 | 2.61 | 2.90 | 24.5 | 4.54 |
| classifier | 2.02 | 0.363 | 0.070 | 0.108 | 0.126 | 0.740 | 0.199 |
| few classes | 2.92 | 1.09 | 0.193 | 0.319 | 0.308 | 2.84 | 0.605 |
| rows | 4.39 | 0.892 | 0.193 | 0.258 | 0.309 | 3.92 | 0.835 |
| segmentation | 30.0 | 5.27 | 0.903 | 1.33 | 1.49 | 16.1 | 3.16 |
| 2^16 elements | 0.269 | 0.050 | 0.050 | 0.056 | 0.056 | 0.091 | 0.043 |
| 2^10 elements | 0.0043 | 0.0011 | 0.0010 | 0.0011 | 0.0011 | 0.002 | 0.015 |

In place `spawn` and `ka` with 0.9.45 take the same time as out of place, but for
`rows`, 20% slower. With 0.10.0 the kernel varied between runs in place: 3.0, 3.6 and
4.7 ms on `attention` in three of them.

The loops, in nanoseconds per element as above:

| loop | Julia | PoCL | PoCL ÷ Julia |
|---|--:|--:|--:|
| `exp`, with `round` | 0.549 | 1.500 | 2.7× |
| `exp`, adding and subtracting a constant | 0.583 | 0.873 | 1.5× |
| `max`, `@fastmath` | 0.047 | 1.790 | 38× |
| `max`, `@fastmath`, entries clamped (NNlib) | 0.062 | 1.128 | 18× |
| `max`, on the bit patterns | 0.081 | 0.092 | 1.1× |
| sum | 0.047 | 0.061 | 1.3× |
| scaling | 0.074 | 0.092 | 1.2× |
| softmax of a slice, as in `ka_softmax.jl` | 0.812 | 1.094 | 1.3× |
| the same in place, array given once | 0.834 | 1.091 | 1.3× |
| the same in place, array given twice | 0.834 | 3.889 | 4.7× |

## Takeaways

- **The loops are worth more than the threads.** On one thread they are 11–15× faster
  than the broadcasts on the Threadripper (57.9 → 3.8 ms on `attention`, 1.49 → 0.10 on
  `classifier`) and 5–6× on the M1; on slices of 10 elements, 2.7× on both. The
  broadcasts call the `exp` of Base, which the compiler does not vectorize, and go
  through the array three times.

- **Tasks add 5–18× on the Threadripper, and 4.6–6× on the M1**, wherever the memory
  is not the limit. `attention` and `long slices` are 64 MB to read and 64 MB to write: on
  the Threadripper they gain 2.3× out of place, and 11–15× in place.

- **With KernelAbstractions 0.9 the kernel is tasks under another name.** In `Float32`
  it is within 20% of `spawn` on the Threadripper, apart from `segmentation` (1.4×) and
  from the two largest arrays in place (1.5× and 2.7×), and it is 1.3–1.7× slower on the
  M1. Its tasks get a fixed share of the work each: the two efficiency cores of the M1
  hold the others back, where the tasks of `spawn` take chunks from each other's ranges.
  Part of the gap is the kernel being written for PoCL: its loops are about 10% slower
  than NNlib's when Julia compiles them. It is ahead in one case, `rows` in `Float64`
  on the Threadripper, by 25%.

- **With KernelAbstractions 0.10 the kernel is 1.5–2.5× slower than the tasks** in
  `Float32` on the Threadripper wherever the memory is not the limit (`classifier`,
  `few classes`, `rows`, `segmentation`, and everything in place), and 1.4–1.8× on the
  M1. PoCL's code
  is that much slower on one thread already: 2.2× on a slice on the Threadripper, where
  it uses 256-bit vectors and separate multiplications and additions, against Julia's
  512-bit vectors and fused multiply-adds, and 1.3× on the M1, where the multiply-adds
  are not fused either. Its threads scale as well as the tasks do.

- **Small arrays are a tie, because neither starts a thread.** `spawn` and `ka` run
  their loops directly below 2^17 elements. The kernels that do launch on a small array
  pay for it: 4 µs on 1024 elements with 0.10.0, against 0.8.

- **The first call costs 3–4 s with PoCL**, against 0.6 s for the tasks.

- **Against PyTorch on the Threadripper**, `spawn` is 2.5× faster on `attention` and
  `long slices`, 1.8× on `rows`, 4.5× on `segmentation` and 6× on `few classes`, and
  1.3× slower on `classifier` (19 against 15 µs). On one thread the loops are 1.5–8×
  faster than PyTorch's from 2^16 elements up. PyTorch is ahead at 2^16 elements, 5
  against 18 µs, where it uses its threads and NNlib does not: after a few milliseconds
  without work Julia's threads are asleep, and waking them up took 15–25 µs here, more
  than the work itself. On the M1 `spawn` is 2–4× faster than PyTorch from `classifier`
  up, and the two are level at 2^16 elements (50 against 43 µs).

### What the kernel needs on PoCL

The annotations Julia's compiler goes by do not reach PoCL, so the kernel is not the
loops of NNlib moved into `@kernel`:

- **No `@fastmath` reduction.** A `@fastmath max` over a slice stays scalar, 35× slower
  than in Julia. The kernel takes the maximum of the bit patterns, as integers, which is
  vectorized everywhere. The sum is vectorized as it is.

- **No `round` on some CPUs.** It becomes OpenCL's `rint`, which PoCL does not vectorize
  on the M1: the `exp` of NNlib is 2.7× slower there, 1.5× with the rounding done by
  adding and subtracting a constant. On the Threadripper the two are the same.

- **A kernel of its own in place.** `@simd ivdep` is lost, and PoCL does not vectorize a
  loop reading one argument and writing another when the two are the same array: 17×
  slower on the Threadripper. A kernel taking the array once is as fast as out of place.

- **No fused multiply-add**, on either CPU, and half the vector width on AVX-512. These
  are what is left of the gap, and the kernel cannot do anything about them. (The
  attention benchmark found that `POCL_DEBUG_LLVM_OPTS=force-vector-width=16` gives the
  width back.)

### Workgroups, and KernelAbstractions.jl#890

- **#890 is about loops with the same bounds in every work-item.** `ka-unsafe-1n`, whose
  work-items loop over `1:n`, is 3.6× slower than the masked `ka-slice` on `attention`
  with 0.10.0 (6.1 against 1.7 ms), 2.8× on `long slices` and 1.5× on `classifier`.
  `ka-unsafe` is the same kernel with the loops over the indices of the slice,
  `off+1:off+n`, which differ from one work-item to the next: it is as fast as the
  masked one. Here PoCL moved a loop outside the one over the work-items only when its
  bounds did not depend on them.

- **A workgroup size of 1 avoids it by construction**, which is what `ka` does, with a
  chunk of slices for each work-item so that the workgroups are few. One slice for each
  work-item and a workgroup of 1 (`ka-slice-1`) costs 10–30%.

- **With 0.9 the workgroup size has to be given.** The default puts up to 1024
  work-items in a workgroup, and a workgroup runs on one task: the 512 slices of
  `classifier` run on one thread (0.105 against 0.019 ms). And a workgroup of 1 on a
  small array starts a task for every thread: 11 µs for 1024 elements, against 0.9.

- **`unsafe_indices` needs full workgroups.** Without the mask the work-items past the
  end of the range run as well, and write past the end of the array: the benchmark
  only runs these kernels when the number of slices is a power of two.

### For NNlib

`softmax!` and `logsoftmax!` of `Array`s take the loops on Julia tasks. With one
exception among these measurements they are level with the kernel or ahead of it, by
up to 2.7×; they need no second formulation of the loops for another compiler; they
honor `NNlib.@disallow_spawns`; and they compile in a fraction of the time. A kernel
written for a GPU would be a different one again, with a work-item for each slice or
each element.
