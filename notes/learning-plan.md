# HPC Learning Plan (Oct 5 – Dec 31, 2026)

Goal: apply for HPC roles by the end of 2026.
Budget: ~16 h/week (weekday evenings + weekends).
Many internship / national-lab deadlines are rolling in the fall:
**minimum PETSc project + updated résumé by mid-November, then start applying.**

Develop in WSL; run scaling experiments on the CRC cluster.

## Project plan

- [x] **Week 1 (Oct 5–11)** — Install PETSc in WSL, run KSP tutorial `ex2`, get a CRC account
- [ ] **Week 2 (Oct 12–18)** — Repository + CMake finding PETSc (pkg-config, `mpicc`/`mpicxx`); "hello PETSc" printing ranks with `mpiexec -n 4`; optional: read about DMDA
- [ ] **Week 3 (Oct 19–25)** — 2D Poisson solver with DMDA; verify with a manufactured solution (error should decrease like h²)
- [ ] **Week 4 (Oct 26–Nov 1)** — Compare solvers and preconditioners (`-ksp_type`, `-pc_type`, incl. `gamg`; `-log_view`)
- [ ] **Weeks 5–6 (Nov 2–15)** — On CRC: batch jobs, strong and weak scaling, plots
- [ ] **Week 7 (Nov 16–22)** — My Eigen GMRES vs. PETSc's KSPGMRES; README → **minimum version done, update résumé, start applying**
- [ ] **Weeks 8–10 (Nov 23–Dec 13)** — OpenMP project: CSR SpMV, roofline analysis, profiling
- [ ] **Weeks 11–12 (Dec 14–31)** — Interview preparation, applications; optional first look at CUDA/Kokkos

## Weekly reading (1–2 h per week)

How to make it count:
- write a few lines of notes in your own words after each session (below);
- practice explaining each topic out loud in about 1–2 minutes;
- connect each topic to your own project data.

| Week | Topic | Where to read | Done |
|------|-------|---------------|------|
| 1 | Amdahl's and Gustafson's laws | LLNL *Introduction to Parallel Computing* | [x] |
| 2 | Memory hierarchy, caches, cache lines, locality | Eijkhout (single-processor chapter); Drepper (CPU caches, what programmers can do) | [x] |
| 3 | MPI basics: processes, ranks, send/receive, deadlock | LLNL MPI tutorial | [ ] |
| 4 | MPI collectives (`Allreduce`, `Bcast`); dot products as a Krylov bottleneck | LLNL MPI tutorial; Eijkhout (parallel linear algebra) | [ ] |
| 5 | Strong vs. weak scaling, parallel efficiency, load balancing | LLNL intro; Hager & Wellein | [ ] |
| 6 | Domain decomposition, halo/ghost exchange (what DMDA does) | Eijkhout; PETSc manual (DMDA) | [ ] |
| 7 | Preconditioning: Jacobi, ILU, multigrid; why multigrid scales | Saad (preconditioning chapters); ATPESC talks | [ ] |
| 8 | Shared memory and OpenMP: threads, `parallel for`, scheduling | LLNL OpenMP tutorial | [ ] |
| 9 | Race conditions, false sharing, synchronization costs | LLNL OpenMP tutorial; Drepper | [ ] |
| 10 | Roofline model, arithmetic intensity, memory- vs. compute-bound | Roofline paper; Hager & Wellein | [ ] |
| 11 | GPU basics: SIMT, memory transfers, when GPUs help | ATPESC GPU talks | [ ] |
| 12 | Review: explain each topic out loud | My own notes | [ ] |

## References

**Core (free)**
- Victor Eijkhout, *Introduction to High Performance Scientific Computing* (free PDF, TACC) — main reference
- LLNL HPC tutorials: *Introduction to Parallel Computing*, *MPI*, *OpenMP*
- Ulrich Drepper, *What Every Programmer Should Know About Memory* (free PDF)
- Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *Communications of the ACM*, 2009

**Books for depth**
- Hager & Wellein, *Introduction to High Performance Computing for Scientists and Engineers*
- Peter Pacheco, *An Introduction to Parallel Programming*
- Victor Eijkhout, *Parallel Programming in MPI and OpenMP* (free)

**Videos**
- ATPESC (Argonne Training Program on Extreme-Scale Computing) lectures on YouTube

**For the PETSc project**
- PETSc user manual and tutorials (petsc.org)
- Yousef Saad, *Iterative Methods for Sparse Linear Systems* (free online)

## My notes

### Week 1 — Amdahl and Gustafson

- Amdahl: S(p) = 1 / (s + (1 − s)/p). With s = 0.1: max speedup 10; p = 100 gives only ≈ 9.2. → strong scaling.
- Gustafson: serial time fixed, parallel work grows with p: S(p) = s + p(1 − s). With s = 0.1, p = 100: ≈ 90. → weak scaling.
- In a Krylov solver, MatMult and vector updates parallelize well (local work + neighbor exchange), but dot products/norms need a global `MPI_Allreduce` → acts like a serial part. That's why PETSc's GMRES uses classical Gram-Schmidt by default (one Allreduce per step instead of j). Block Jacobi weakens with more processes → more iterations.

### Week 1 — `ex2.c`

- Structure: `PetscInitialize` → create Mat/Vec/KSP → assemble → solve → destroy everything → `PetscFinalize`. Every Create/Duplicate needs a Destroy.
- `MatGetOwnershipRange`: each process owns a contiguous block of rows [Istart, Iend) and only fills its own rows.
- Assembly: values set for another process's rows are stashed and sent in `MatAssemblyBegin/End`; assembly also finalizes the internal structure (e.g. what MatMult needs from neighbors). Begin/End split allows overlapping communication with computation.
- Testing: choose exact u = 1, compute b = A u, solve, report ‖x − u‖ (manufactured solution).
- `KSPSetFromOptions` reads the options database at run time and is called after the code's defaults, so command-line options win.
- `PetscCall`: checks the error code; on error prints a traceback and returns.

### Week 2 — Caches and locality

Questions:
1. What is a cache line, and why does reading one `double` load 64 bytes?
2. Spatial vs. temporal locality — one example of each from my own code.
3. Eigen is column-major: why does the loop order `for i … for j … A(i, j)` matter?
4. In a 5-point-stencil matrix-vector product, each x value is used up to 5 times. When does that reuse come from cache, and when from main memory?

Notes (my laptop: Intel Core Ultra 7 155U — 64 B lines, L1d 48 KiB/core, L2 2 MiB/core, L3 12 MiB shared):

1. **Cache line** = the smallest unit moved between memory and cache (64 B = 8 doubles). Reading one `double` loads the whole aligned 64-byte line containing it. Why: memory latency is high (~100 ns), so the hardware pays it once and brings neighbors along, betting that I will use them (spatial locality). If I only use 8 of the 64 bytes, I waste 7/8 of the bandwidth.
2. **Spatial locality**: using addresses close to each other, e.g. `for (i...) sum += x[i]*y[i]` — `x[i+1]` is already in the line loaded for `x[i]`.
   **Temporal locality**: using the *same* data again soon, while it is still in cache. Example from my GMRES (modified Gram-Schmidt): `w` is reused for every `h(i,j) = v_i · w; w -= h(i,j) v_i`, i = 0..j — if `w` fits in cache, it is loaded from memory once and reused j times. (The accumulator `sum` in a dot product is also temporal, but it lives in a register.)
3. **Column-major**: `A(i,j)` and `A(i+1,j)` are adjacent; `A(i,j)` and `A(i,j+1)` are `n` doubles (8n bytes) apart. With `j` as the inner loop, every access touches a new cache line and uses only 8 of its 64 bytes; for large n those lines are evicted before `i` advances, so each is reloaded → up to 8× more memory traffic. Fix: `for j ... for i ... A(i,j)` (inner loop runs down a column, stride 1). (Row-major C arrays: the opposite.)
4. **5-point stencil** on an N×N grid, natural (row-by-row) ordering: `y_k = 4x_k − x_{k−1} − x_{k+1} − x_{k−N} − x_{k+N}`. The value `x_m` is used by rows m−N, m−1, m, m+1, m+N.
   - `x_{k±1}`: same/neighboring cache line, used almost immediately → always from cache.
   - `x_{k±N}`: first use and last use of `x_m` are 2N rows apart, so about 2N doubles (16N bytes) of `x` must stay in cache in between.
     - Fits in cache → each `x` comes from main memory ~once. On my laptop: L1 (48 KiB) for N ≲ 3000, L2 (2 MiB) for N ≲ 130 000.
     - Doesn't fit (very wide grid rows) → `x_{k−N}` is evicted before reuse and loaded again from memory (up to 3 times per value).
   - The bigger issue: an assembled CSR matrix has 5 values + 5 column indices per row ≈ 60 B/row, streamed once with **no reuse**, vs. 16 B/row for `x` and `y`. So matrix traffic dominates → SpMV is **memory-bound** (→ roofline, week 10). A matrix-free stencil (computing the 4/−1 coefficients on the fly) removes that traffic.
   - In parallel with DMDA, N is the *local* grid width, so the working set per rank is even smaller.


### Week 2 — DMDA basics (prep for week 3)

Questions:
1. What do `DMDACreate2d`'s arguments mean?
2. Global vs. local vector — why `DMGlobalToLocal`?
3. What does `DMDAGetCorners` give, and why loop over `xs..xs+xm` instead of `0..N`?
4. How does `MatSetValuesStencil` avoid computing global row numbers?

Notes:

1. **`DMDACreate2d`**
   - Boundary type per direction: `DM_BOUNDARY_NONE` (domain just ends), `DM_BOUNDARY_PERIODIC` (wraps around), `DM_BOUNDARY_GHOSTED` (extra ghost points outside the physical domain). Dirichlet Poisson → `NONE`.
   - Stencil type: `DMDA_STENCIL_STAR` = only axis neighbors (5-point Laplacian); `DMDA_STENCIL_BOX` = also diagonals (9-point, mixed derivatives, Q1 FEM). Poisson → **STAR, width 1**.
   - `M, N`: global grid size. `m, n`: process grid, usually `PETSC_DECIDE`.
   - `dof`: unknowns per grid point — 1 for scalar Poisson, 4 for 2D Euler (ρ, ρu, ρv, E).
   - Stencil width: how far the stencil reaches, e.g. 2nd-order central difference (u_{i+1} − 2u_i + u_{i−1})/Δx² → 1; 4th-order central → 2. It sets the **number of ghost layers** per rank and the **nonzero pattern** `DMCreateMatrix` preallocates.
2. **Global vs. local vector**: a global vector holds only the owned DoFs and is the parallel vector KSP works on. A local vector belongs to one process and holds owned + ghost values. `DMGlobalToLocal` fills the ghosts from neighbors (halo exchange). Needed when I compute with neighbor values myself (matrix-free operators, residuals) — not for assembly with `MatSetValuesStencil` (→ week 6 reading).
3. **`DMDAGetCorners`**: `xs, ys` = global indices where my part starts; `xm, ym` = **how many** points I own in each direction. Loop `i = xs … xs+xm−1`, `j = ys … ys+ym−1`. Since `i, j` are global grid indices, I can compute coordinates (`x = i*h`) and detect boundaries (`i == 0 || i == M−1`) directly. Not `0..N` because each rank only sets the rows it owns. (`DMDAGetGhostCorners` gives the region including ghosts.)
4. **`MatSetValuesStencil`**: rows and columns are given as grid positions `MatStencil {i, j}` (plus `c` for the component when dof > 1), not global row numbers. This matters because **PETSc's parallel ordering is not natural ordering** — each rank's rows are numbered contiguously, so `row = j*M + i` is right on 1 process and wrong on 4. For an interior point: one row stencil `{i, j}`, five column stencils (itself + 4 neighbors), values 4, −1, −1, −1, −1 (× 1/h²). Column stencils may point into a neighbor's region — allowed thanks to the ghost layer.

### Step 2.1 — `src/dmda.cpp` (print each rank's corners)

Goal: create an 8×8 DMDA, print each rank's `xs, ys, xm, ym`, run with 1/2/4 ranks.

Lessons from the code review:
- **Uninitialized variables**: declaring `xs, ys, xm, ym` without calling `DMDAGetCorners` prints garbage. Compiler warning `-Wmaybe-uninitialized` caught it → treat warnings as bugs.
- **`DMDAGetCorners` has 7 arguments**: `(da, &x, &y, &z, &m, &n, &p)` — starts first, then widths. In 2D pass `NULL` for `z` and `p`. Check the signature in the man page instead of guessing.
- **Every Create needs a Destroy**: `DMDestroy(&da)` before `PetscFinalize`. Check leaks with `-malloc_dump`.
- `PetscFunctionBeginUser;` at the top of `main` (better error tracebacks).
- Print `PetscInt` with `"%" PetscInt_FMT`, not `%d` (PetscInt may be 64-bit).
- Print the rank in each line so output from different ranks is easy to tell apart; one line per rank is more readable than several prints.
- Optional: `DMSetFromOptions(da)` before `DMSetUp` → grid size from the command line (`-da_grid_x 16 -da_grid_y 16`).
- **CMake arguments are separated by spaces, not commas**: `target_compile_options(dmda PRIVATE -Wall -Wextra)`. With a comma, the target name becomes `dmda,`.
- New executable → add `add_executable` + `target_link_libraries` in `CMakeLists.txt`, otherwise it is never built.

Build and run (from the project root):
```bash
cmake -S . -B build          # configure (needed again only for new files/targets)
cmake --build build          # compile; enough after editing .cpp files
mpiexec -n 1 ./build/dmda
mpiexec -n 4 ./build/dmda
mpiexec -n 4 ./build/dmda -malloc_dump   # no extra output = no leaks
```
When a build fails, read the **first** error — later ones are often caused by it.

TODO (next session):
- [ ] Fix the two build errors: commas in `target_compile_options`; `DMDAGetCorners` needs 7 arguments
- [ ] Build and run with `-n 1`, `-n 2`, `-n 4`
- [ ] Check: `-n 1` gives `xs 0, ys 0, xm 8, ym 8`; with `-n 4` the `xm × ym` values sum to 64 (no gaps/overlaps); how are the 4 ranks arranged (2×2 or 4×1)?
- [ ] Run with `-malloc_dump`
- [ ] Commit, then step 2.2: assemble the 5-point Laplacian with `DMCreateMatrix` + `MatSetValuesStencil`
