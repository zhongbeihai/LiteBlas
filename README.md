# CSE260: Optimized DGEMM Kernel

This project implements a blocked `dgemm` for ARMv8.2-A + SVE platforms. It follows the classic *pack–macro kernel–micro kernel* structure: panels of `A` and `B` are packed into contiguous buffers, a five-level blocking loop orchestrates cache reuse, and an 8×4 SVE microkernel performs the innermost compute.

- Build: `make` (produces `mm`)
- Run demo: `./mm --m <rows> --k <k> --n <cols>`
- Key dirs: `cse260_hw1/` (kernel + packing), `matrix/`, `utils/`



## Packing Overview

Packing converts strided matrix panels into dense, cache-aligned micro-panels that the microkernel can load without gather/scatter operations.

- `pack_A_panel_MrKc`: copies an `ib × pb` tile from `A[ic:ic+ib, pc:pc+pb]`, storing it as `pb` consecutive blocks of size `Mr`. Each block contains a column of the source tile laid out contiguously in memory. Tails smaller than `Mr` are zero-padded to simplify the microkernel.
- `pack_B_panel_KcNr`: copies a `pb × jb` tile from `B[pc:pc+pb, jc:jc+jb]`, producing `pb` rows, each padded/truncated to `Nr` elements. This layout matches the SVE microkernel’s expectation of contiguous `Nr`-wide vectors.
- Both routines rely on `aligned_malloc` (64-byte alignment) to ensure efficient cache and SVE loads.

Packing amortizes strided access costs: the slow, irregular loads are performed once per macro-tile, while the microkernel performs only unit-stride accesses during accumulation.


## Five-Level Loop Nest

`DGEMM_mykernel::my_dgemm` implements the canonical five-loop blocking strategy. Each loop targets a different level of the memory hierarchy:

1. **`jc` loop (NC)** – sweeps across the output matrix in column panels that fit in the LLC, keeping packed `B` in cache.
2. **`pc` loop (KC)** – slices the shared dimension so the packed `A`/`B` panels fit in the L2 cache.
3. **`ic` loop (MC)** – iterates over row panels of `A`, exposing rows of `C` that will be updated by the macro kernel while keeping packed `A` resident.
4. **`i` loop (MR)** – walks consecutive micro-panels of `A`; each step selects an `Mr × Kc` block already in `packA`.
5. **`j` loop (NR)** – selects the matching `Kc × Nr` micro-panel from `packB`, calling the microkernel to update an `Mr × Nr` tile of `C`.

This organization keeps data hot at the appropriate cache level and reduces TLB pressure by reusing packed buffers before moving on.


## Microkernel Fundamentals

`my_dgemm_sve_8x4` is the inner compute kernel. It assumes an `8 × 4` tile (`param_mr = 8`, `param_nr = 4` after tuning) and performs:

1. **Prologue** – loads eight rows of `C` into SVE registers (`c0`–`c7`), one vector per row.
2. **Main loop** – for each `k`:
   - Loads one `Nr`-wide vector from packed `B`.
   - Broadcasts the corresponding scalar from packed `A`.
   - Performs FMA (`svmla_f64_x`) to accumulate into `c0`–`c7`.
3. **Epilogue** – stores the updated vectors back to `C`.



## Parameters and Tuning

Tuning knobs are centralized in `cse260_hw1/parameters.h`:

- `PARAM_MC`, `PARAM_NC`, `PARAM_KC` – block sizes for the three outer loops.
- `PARAM_MR`, `PARAM_NR` – microkernel tile sizes.

You can override these at compile time (e.g., `make MY_OPT="-DPARAM_MR=8 -DPARAM_NR=4"`). When changing microkernel dimensions, adjust both the packing layout and the SVE kernel accordingly.

---

## Testing and Profiling

- Functional check: `./mm --m 128 --k 128 --n 128 --validate`.
- Baseline comparison: link against OpenBLAS via the provided harness (`make && ./mm --compare`).
- Performance hints: inspect cache misses with `perf stat` (if available) and experiment with block sizes to balance reuse and packing cost.

---

## File Guide

- `cse260_hw1/dgemm_mykernel.cpp` – driver, blocking logic, macro kernel.
- `cse260_hw1/kernel.cpp` – reference kernels and SVE microkernel.
- `cse260_hw1/pack.cpp` / `pack.h` – aligned allocation and packing routines.
- `matrix/`, `utils/` – matrix utilities, timers, CLI support.

