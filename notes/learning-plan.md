# HPC Learning Plan (Oct 5 – Dec 31, 2026)

Goal: apply for HPC roles by the end of 2026.
Budget: ~16 h/week (weekday evenings + weekends).
Many internship / national-lab deadlines are rolling in the fall:
**minimum PETSc project + updated résumé by mid-November, then start applying.**

Develop in WSL; run scaling experiments on the CRC cluster.

## Project plan

- [x] **Week 1 (Oct 5–11)** — Install PETSc in WSL, run KSP tutorial `ex2`, get a CRC account
- [x] **Week 2 (Oct 12–18)** — Repository + CMake finding PETSc (pkg-config, `mpicc`/`mpicxx`); "hello PETSc" printing ranks with `mpiexec -n 4`; optional: read about DMDA
- [x] **Week 3 (Oct 19–25)** — 2D Poisson solver with DMDA; verify with a manufactured solution (error should decrease like h²) — DONE early (2026-10-18)
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
| 3 | MPI basics: processes, ranks, send/receive, deadlock | LLNL MPI tutorial | [x] |
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

TODO (done 2026-10-14):
- [x] Fix the two build errors: commas in `target_compile_options`; `DMDAGetCorners` needs 7 arguments
- [x] Build and run with `-n 1`, `-n 2`, `-n 4`
- [x] Check: `-n 1` gives `xs 0, ys 0, xm 8, ym 8`; with `-n 4` the `xm × ym` values sum to 64 (no gaps/overlaps); how are the 4 ranks arranged (2×2 or 4×1)?
- [x] Run with `-malloc_dump`
- [x] Commit, then step 2.2: assemble the 5-point Laplacian with `DMCreateMatrix` + `MatSetValuesStencil`

Results (8×8 grid):
```
 -n 2: 1×2 process grid (split in y)     -n 4: 2×2 process grid
 ┌─────────────────┐                     ┌────────┬────────┐
 │ Rank 1  ys=4    │                     │ Rank 2 │ Rank 3 │   ys=4
 ├─────────────────┤                     ├────────┼────────┤
 │ Rank 0  ys=0    │                     │ Rank 0 │ Rank 1 │   ys=0
 └─────────────────┘                     └────────┴────────┘
   xm=8, ym=4                              xs=0     xs=4,  xm=ym=4
```
- Ranks are numbered x first (like grid points: i is fastest).
- PETSc ordering with `-n 4`: rank 0 → rows 0–15, rank 1 → 16–31, … So point (4,0) is row 16, not `j*8+i = 4` → use `MatSetValuesStencil`, never compute row numbers myself.
- **Why squares, not strips?** Same work per rank (N²/p), but ghost values per rank: strips 2N (constant in p), squares ≈ 4N/√p (shrinks). For 8×8, p=4: worst rank 16 (strips) vs 8 (squares). Strips have fewer neighbors (fewer messages, less latency) — only better for small p. Force a layout: `-da_processors_x 1 -da_processors_y 4` (needs `DMSetFromOptions`).

### Step 2.2 — `src/poisson.cpp` (assemble the Dirichlet 5-point Laplacian)

Problem: −Δu = f on (0,1)², **u = 0 on the boundary (homogeneous Dirichlet)**. Manufactured solution u = sin(πx)sin(πy), f = 2π² sin(πx)sin(πy).

How the matrix is built (M×M points incl. boundary, h = 1/(M−1), 1 process: row k = j*M + i):
- One unknown per grid point → one row (equation) + one column per point.
- **Boundary row**: `1 · u = 0` → one entry, 1 on the diagonal (pass `&row` as the column too).
- **Interior row**: (4u − 4 neighbors)/h², but **only interior neighbors** — a boundary neighbor has u = 0, so its term drops (if g ≠ 0: move g/h² to the RHS). This keeps A **symmetric** → CG works.
- Entries per row on 5×5: center 5, next to one edge 4, next to a corner 3, boundary 1.
- Loop over owned points (`j` outer, `i` inner), fill `col[]`/`v[]` with a counter `num` (= next free slot), call `MatSetValuesStencil(A, 1, &row, num, col, v, INSERT_VALUES)`, then `MatAssemblyBegin/End`.
- Dirichlet vs Neumann: Neumann (ex50) has row sums 0 → A·1 = 0 → singular, needs `MatNullSpace` + compatibility ∫f = 0. **Check what problem an example solves before copying its code.**

Bugs I fixed (in order):
1. `Mat A;` not declared → compile error.
2. Typo `PestcCall` → the compiler suggests the fix; read it.
3. `M, N` uninitialized → garbage grid size. **No compiler warning this time** — can't rely on warnings.
4. `MatNullSpace` declared but never created → "Wrong type of object: Parameter # 2". Read the traceback **bottom up** to find my line.
5. Neumann boundary rows copied from ex50 → replaced with identity rows.
6. Missing `num++` after the center → next entry overwrote it, diagonal = 0. **Every filled slot must advance `num`.**
7. Off-by-one: right/top neighbor checked `i + 1 < M` instead of `i + 1 < M − 1` → boundary columns added. Most common bug in numerical code.
Also: `h` should be `PetscReal`, not `PetscScalar` (complex builds); scale interior entries by 1/h², boundary rows stay 1.

Reading `-mat_view`:
- Prints at **every** assembly → the first all-zero matrix comes from `DMCreateMatrix` (reserved pattern).
- `(col, 0.)` entries are **stored zeros** from the preallocated 5-point pattern — ignore them, look only at nonzeros.
- Check by hand on a tiny grid (M = 5, h = 0.25 → 64 and −16), and in code with `MatIsSymmetric(A, 1e-12, &isSym)` on `-n 1` and `-n 4` → both 1.

### Step 2.3 — right-hand side, solve, verify O(h²) (done 2026-10-18)

Code steps:
- Vectors: `DMCreateGlobalVector(da, &b)` (same parallel layout as A's rows), `VecDuplicate(b, &x)` / `&u`. Every Create → Destroy.
- Fill with grid indices: `DMDAVecGetArray(da, b, &barr)` → loop over owned points, **`barr[j][i]`** (j first!) → `DMDAVecRestoreArray`. Boundary entries of b = 0 (equation `1·u = 0`); interior = f.
- Solve: `KSPCreate` → `KSPSetOperators(ksp, A, A)` → `KSPSetFromOptions` → `KSPSolve(ksp, b, x)` → `KSPDestroy`. Only after `MatAssemblyEnd`.
- Error: `VecAXPY(x, -1.0, u)` (x ← x − u), `VecNorm(x, NORM_INFINITY, &err)`.
- Grid size from the command line: `PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL)` before `DMDACreate2d`, then `N = M`.

Bugs / lessons:
- **Segfault**: wrote `barr[j][i]` *before* `DMDAVecGetArray` → garbage pointer → SIGSEGV. For a segfault, the traceback line is only "the last `PetscCall` that succeeded" — the bug is somewhere after it.
- `PetscCosReal` instead of `PetscSinReal` for y → caught by checking one value by hand (center should be 2π² ≈ 19.739).
- `PetscINT_FMT` → names are **case-sensitive**; one typo gave 3 compiler messages — read the first.
- `"%d" PetscInt_FMT` printed `5d` — `PetscInt_FMT` already contains the `d`; write `"%" PetscInt_FMT`.
- PETSc warns about **unused options** (`Option left: -vec_view`) → catches spelling mistakes like `-ksp_tpye`.
- Use `-ksp_rtol 1e-12` for convergence studies so solver error doesn't pollute discretization error.

Solver observations (M = 5):
- Default (GMRES + ILU) and CG: 4 iterations on 1 process. CG works because A is SPD (symmetric boundary treatment!).
- CG on 4 processes: 5 iterations — default PC becomes block Jacobi (ILU per rank, ignores coupling between ranks) → weaker → more iterations.
- With 5×5 on a 2×2 process grid, blocks are 3×3, 2×3, 3×2, 2×2 (5 not divisible by 2).

**Convergence result** (`-ksp_type cg -ksp_rtol 1e-12`, max-norm error):

| M | h | error | ratio |
|---|---|---|---|
| 5 | 0.25 | 5.303e-2 | — |
| 9 | 0.125 | 1.295e-2 | 4.095 |
| 17 | 0.0625 | 3.219e-3 | 4.023 |
| 33 | 0.03125 | 8.036e-4 | 4.006 |
| 65 | 0.015625 | 2.008e-4 | 4.001 |
| 129 | 0.0078125 | 5.020e-5 | 4.000 |

- Ratio → 4 ⇒ **O(h²)**. Early ratios > 4 because of higher-order terms (h⁴, …) on coarse grids.
- `-n 1` and `-n 4` give the same error (5.02009e-05 at M = 129) → parallel code matches serial.
- **Analytic check**: sin(πx)sin(πy) is an eigenvector of the discrete Laplacian with λ_h = (8/h²) sin²(πh/2), so the discrete solution is (2π²/λ_h)·u and the max error is exactly 2π²/λ_h − 1. Every measured error matches this formula to all printed digits.

### Week 3 — MPI basics

Questions:
1. What is an MPI process, a rank, a communicator? `PETSC_COMM_WORLD` vs `MPI_COMM_WORLD`?
2. Blocking vs non-blocking communication — why does non-blocking help?
3. How can two processes deadlock with `MPI_Send`/`MPI_Recv`? How to avoid it?
4. In my 4-rank Poisson run, which PETSc calls communicate, and what kind of communication?

Notes:

1. **Process**: one running copy of the program with its **own private memory** (distributed memory). Processes share no variables — data moves only by explicit messages.
   **Rank**: the process's integer ID inside a communicator, `0 … size−1` (`MPI_Comm_rank`, `MPI_Comm_size`).
   **Communicator**: a group of processes that can talk to each other, plus a private "context" so messages in one communicator never mix with another. `MPI_COMM_WORLD` = all processes started by `mpiexec`.
   `PETSC_COMM_WORLD`: PETSc's communicator — by default a **duplicate** of `MPI_COMM_WORLD` (same processes, separate context, so PETSc's internal messages can't collide with mine). It can be set to a subset of processes before `PetscInitialize`.
2. **Blocking** (`MPI_Send`/`MPI_Recv`): the call returns only when it is safe to continue — for `MPI_Recv` the data has arrived; for `MPI_Send` the send buffer can be reused (the message may only be buffered, not yet received).
   **Non-blocking** (`MPI_Isend`/`MPI_Irecv`): returns immediately with a request; I must not touch the buffer until `MPI_Wait`/`MPI_Waitall`. In between I can do **independent** work → overlap communication with computation.
   Ideal cost: blocking T = T_comp + T_comm; overlapped T ≈ max(T_comp, T_comm) (upper bound on the gain — needs work that doesn't depend on the incoming data).
   PETSc example: parallel `MatMult` starts the halo exchange, multiplies the **local (diagonal) block** while ghost values are in flight, then finishes the exchange and multiplies the **off-process block**. Same idea as `MatAssemblyBegin/End` and `DMGlobalToLocalBegin/End`.
3. **Deadlock**: every process waits for another, forever.
   - Rank 0 and rank 1 both call `MPI_Recv` first, then `MPI_Send` → both wait for a message nobody sends. Always deadlocks.
   - Both call `MPI_Send` first → **unsafe**: works for small messages (MPI buffers them, "eager" protocol) but **hangs for large ones** (send waits for the matching receive, "rendezvous"). Bug that appears only when the problem size grows.
   - Fixes: `MPI_Sendrecv` (MPI pairs them safely); order by rank (even ranks send first, odd ranks receive first); or non-blocking `MPI_Irecv` + `MPI_Isend` + `MPI_Waitall`.
4. **Communication in my Poisson solver** (4 ranks):
   - `MatAssemblyBegin/End`: sends values set for rows owned by **other** ranks (the "stash"). In my code each rank sets only its own rows, so almost nothing is sent — but the call is still collective (all ranks must call it).
   - `MatMult` (inside every KSP iteration): **neighbor-to-neighbor halo exchange** — each rank needs only the ghost values of `x` along its block edges from adjacent ranks, not the global vector. Cost ∝ block perimeter (→ squares vs. strips).
   - `VecDot` / `VecNorm`: **global reduction** (`MPI_Allreduce`) — every rank computes a partial sum, and all ranks need the total. Involves all ranks → latency grows with log(p) and acts like a serial part (Amdahl).
   - Per CG iteration: 1 `MatMult` (halo) + 2 dot products and a norm (Allreduces). → Week 4 reading: why Allreduce limits Krylov scaling.

### Week 4 plan — compare Krylov solvers and preconditioners

Goal: on a range of grid sizes, measure **iterations** and **time** for several KSP/PC combinations, explain the results, and add them to the README.

- [x] **Step 1 — optimized PETSc build** (~30 min, mostly waiting). Timings from a debug build are meaningless (no `-O3`, extra checks). Keep the debug build for development; add a second build:
  ```bash
  cd ~/petsc
  ./configure PETSC_ARCH=arch-linux-c-opt --with-debugging=0 \
    COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 \
    --with-cc=gcc --with-cxx=g++ --with-fc=gfortran \
    --download-mpich --download-fblaslapack
  make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-c-opt all
  ```
  Then a separate project build dir: `PETSC_ARCH=arch-linux-c-opt cmake -S . -B build-opt && cmake --build build-opt` (add `build-opt/` to `.gitignore`). Question to answer: why does `mpiexec` come from the arch you built with?
- [x] **Step 2 — print the iteration count** in `poisson.cpp` with `KSPGetIterationNumber` (and the converged reason with `KSPGetConvergedReason`), so one output line contains M, iterations, error.
- [x] **Step 3 — learn to read `-log_view`** on one run (e.g. M = 513, CG + GAMG): find `KSPSolve`, `MatMult`, `PCSetUp`, `PCApply`, `VecTDot`/`VecNorm`. What do the columns Count, Time, %T mean? Which event takes most time?
- [x] **Step 4 — experiment 1: iterations vs. grid size** (1 process, opt build), M = 65, 129, 257, 513, 1025:
  - `-ksp_type cg` with `-pc_type none | jacobi | ilu | gamg`
  - `-ksp_type gmres -pc_type ilu` (compare with CG + ILU)
  - Fair comparison: same tolerance for all (`-ksp_rtol 1e-8`) and the same residual norm (`-ksp_norm_type unpreconditioned`; GMRES also needs `-ksp_pc_side right` for that norm).
  - Predict first, then check: κ(A) ∝ h⁻², CG iterations ∝ √κ ∝ 1/h → doubling M doubles iterations for `none`/`jacobi`/`ilu`; GAMG ≈ constant.
- [x] **Step 5 — experiment 2: time vs. grid size** for the same runs: total `KSPSolve` time and setup (`PCSetUp`) time from `-log_view`. Which method wins at small M, which at large M? Why is GAMG's setup expensive but worth it?
- [x] **Step 6 — experiment 3: effect of parallelism on the preconditioner** (M = 513, `-n 1, 2, 4`): CG + ILU (becomes block Jacobi + ILU in parallel) vs. CG + GAMG. How do the iteration counts change, and why?
- [x] **Step 7 — automate**: a small bash script that loops over M and solvers and writes one line per run to a CSV file (`results/week4.csv`).
- [x] **Step 8 — plot** with Python/matplotlib: iterations vs. M and time vs. M (log-log), one line per method. Save as PNG in `results/`.
- [ ] **Step 9 — README**: add a "Solver comparison" subsection with the table/plot and 3–4 bullets explaining it (condition number, why multigrid scales, setup vs. solve cost).
- [ ] **Reading (week 4)**: MPI collectives (`MPI_Allreduce`, `MPI_Bcast`); why dot products limit Krylov scaling. Tie to my `-log_view`: how many `VecTDot`/`VecNorm` calls per CG iteration?

Interview one-liners to be able to say after this week:
- "CG iterations grow like √κ ∝ 1/h for the Laplacian; multigrid makes them h-independent."
- "Block Jacobi weakens as I add processes, because it ignores coupling between subdomains."
- "GAMG has a high setup cost but the lowest total time for large problems."

### Week 4 — steps 1–3 results (2026-10-19)

**Step 1 — opt build**: `arch-linux-c-opt` next to `arch-linux-c-debug` (one PETSc source, several builds via `PETSC_ARCH`).
- Project: `PETSC_ARCH=arch-linux-c-opt cmake -S . -B build-opt` — always use the prefix for `build-opt` commands (CMake re-reads the env var when it re-configures).
- Run with the matching launcher: `MPIOPT=~/petsc/arch-linux-c-opt/bin/mpiexec`. Check linking with `ldd build-opt/poisson | grep petsc`.
- M = 513, CG: debug 4.15 s vs opt 1.69 s → only **2.5×**, because CG is dominated by **memory-bound** MatMult/vector ops; `-O3` speeds up arithmetic, not memory (→ roofline, week 10).

**Step 2 — iterations**: `KSPGetIterationNumber`, `KSPGetConvergedReason` + `KSPConvergedReasons[reason]` (string). Always check the reason — a `DIVERGED_ITS` run has invalid iterations/time.
- Bug: a `)` closed `PetscPrintf(` too early → format warnings + garbage output (`its= -2072633264, reason= (null)`). **Format warnings are real bugs.**
- Habit: save → build → run. If output doesn't change, check the binary is newer than the source (`ls -l`).

**Step 3 — reading `-log_view`** (opt, M = 513, CG + GAMG, rtol 1e-8, 11 iterations, total 0.63 s):
- Columns: **Count** (calls), **Time** (total seconds), **%T** (share of total time), Flop, and for parallel runs Mess/AvgLen/Reduct (messages, reductions).
- Events are **nested**: `KSPSolve` ⊃ `PCApply` ⊃ `MatMult`; `PCSetUp` ⊃ `MatPtAP`, `PCGAMG …`. Also look at **leaf** events (actual work).
- `PCSetUp` 0.376 s (**59%**) > whole `KSPSolve` 0.187 s; `PCApply` = 87% of `KSPSolve`. Largest leaf: `MatMult` 16% time, 45% of flops.
- `MatMult` count 243 ≫ 12 (11 iterations + initial residual): the rest are **inside GAMG** (smoother steps on every level + eigenvalue estimates for the Chebyshev smoother). One GAMG iteration is expensive, but few are needed.
- `VecTDot` 24 / 12 = **2 per CG iteration** (α and β); `VecNorm` ≈ 1 per iteration for CG, the rest from GAMG setup (`VecNormalize`). → ~3 global reductions per CG iteration = 3 `MPI_Allreduce`s in parallel (week 4 reading).

### What is multigrid / GAMG

- **Smoother**: Jacobi/Gauss–Seidel remove **oscillatory** error fast but **smooth** error very slowly (a smooth wave ≈ its local average) → O(1/h²) iterations alone. After a few steps the error is smooth.
- **Coarse-grid correction**: smooth error on a fine grid looks **oscillatory on a coarser grid** → smoothing works again there. Recursively: smooth → restrict residual → (coarser levels) → interpolate correction → smooth. = **V-cycle**.
- Cost of a V-cycle ≈ O(N) (2D: 1 + 1/4 + 1/16 + … ≈ 4/3 of fine-grid work); convergence factor **independent of h** → iterations ~constant as M grows → optimal-order solver.
- **Algebraic** multigrid builds coarse levels from the **matrix alone**: aggregate strongly connected unknowns → interpolation **P**; coarse matrix **A_c = PᵀAP** (Galerkin). Works on unstructured meshes (e.g. DG matrices). Geometric MG would need the coarse grids from me.
- GAMG = PETSc's (smoothed-aggregation) AMG. In `-log_view`: `PCGAMG Squ/Opt lXX` = building aggregates/P on level XX; `MatPtAP` / `PCGAMG Gal` = coarse matrices; extra `MatMult`s = Chebyshev smoother + eigenvalue estimates; `PCApply` = one V-cycle per CG iteration.
- Interview: "Smoothers remove high-frequency error; coarse grids turn low-frequency error into high-frequency error. A V-cycle costs O(N) and converges at an h-independent rate → optimal."
- Read more: Briggs, Henson & McCormick, *A Multigrid Tutorial* (ch. 1–3); Saad ch. 13. (Week 7 reading.)

### Week 4 — steps 4–5 results (2026-10-20)

Setup: opt build, 1 process, `-ksp_rtol 1e-8 -ksp_norm_type unpreconditioned`, M = 65 … 1025.

**Predictions** (CG iterations ∝ √κ):
- Eigenvalues λ(p,q) = (4/h²)(sin²(pπh/2) + sin²(qπh/2)) → λ_min ≈ 2π² (independent of h), λ_max ≈ 8/h² → κ ≈ 4/(π²h²) ∝ 1/h².
- h → h/2 ⇒ κ → 4κ ⇒ √κ → 2√κ ⇒ **iterations ×2** for none / jacobi / ilu.
- Jacobi: P⁻¹A = (h²/4)·A on the interior → same κ → ≈ same as none. ILU: κ smaller by a constant → fewer iterations, still ×2. GAMG: h-independent → ≈ constant.

**Surprise 1 — 1 iteration for none and jacobi.** b = 2π² sin(πx)sin(πy) is an **eigenvector** of A (Ab = λb) → x* = b/λ ∈ span{b} = first Krylov space → CG done in 1 step. Jacobi = constant × A → b still an eigenvector. ILU changes the eigenvectors → normal behavior.
→ Lesson: the manufactured solution is great for **accuracy** checks but a **bad test for solver comparison** (single eigenvector = easiest input). Added `-random_rhs` (`VecSetRandom(b, NULL)`, fixed seed → reproducible); error is only printed for the default problem.

**Iterations (CG, `-random_rhs`):**

| M | none | jacobi | ilu | gamg |
|---|---|---|---|---|
| 65 | 233 | 191 | 63 | 11 |
| 129 | 456 | 379 | 124 | 12 |
| 257 | 924 | 752 | 227 | 13 |
| 513 | 1819 | 1449 | 430 | 15 |
| 1025 | 3647 | 2944 | 877 | 17 |

- none/jacobi/ilu: ×2 per doubling ✓. ILU ≈ 4× fewer than none (same slope, lower line) ✓. GAMG 11 → 17 for 256× more unknowns ✓ (slight growth is typical for AMG).
- Surprise 2: Jacobi ≈ 20% better than none on every grid. Hypothesis: identity boundary rows have eigenvalue 1, an **outlier below** the interior spectrum [2π², 8/h²] without Jacobi; with Jacobi the interior is scaled to [π²h²/2, 2], so 1 is inside → no outlier. Test (optional): zero b on the boundary → none and jacobi should match exactly.

**CG vs GMRES(30), both ILU** (`-ksp_pc_side right` for the unpreconditioned norm): GMRES 74, 197, 823, 2939, **DIVERGED_ITS at M = 1025** (10 000 limit).
- GMRES(30) restarts **every 30 iterations, always** → throws away the Krylov basis → loses optimality, can stagnate. CG gets optimality for free from a short 3-term recurrence (A symmetric) with constant memory.
- A `DIVERGED_ITS` run is **not a result** — mark it "did not converge", never "10000 iterations". (Optional: `-ksp_gmres_restart 200`.)

**Time = PCSetUp + KSPSolve (s):**

| M | none | jacobi | ilu | gamg |
|---|---|---|---|---|
| 129 | 0.033 | 0.034 | 0.027 | 0.041 |
| 257 | 0.318 | 0.304 | 0.233 | 0.150 |
| 513 | 4.26 | 3.43 | 1.87 | 0.658 |
| 1025 | 37.8 | 32.4 | 16.75 | 2.98 |

- Small M: cheap iterations win (ILU fastest; GAMG slowest, setup 68% of its time). Crossover between M = 129 and 257. Large M: GAMG **12.7×** faster than plain CG, 5.6× faster than ILU, even though setup is still ~52% of its time.
- Scaling: none/jacobi/ilu time ∝ iterations × N ∝ (1/h)(1/h²) = **N^1.5** (×8 per doubling of M); GAMG ∝ **N** (×4). Measured GAMG: 3.6, 4.4, 4.5 ✓.
- **Cache effect**: none: ×9.6, **×13.4**, ×8.9. The Mflop/s column (last in `-log_view`) drops 4814 → 4024 → **2357** → 2128. At M = 257 the CSR matrix (~257²·60 B ≈ 4 MB) + vectors fit in L3 (12 MiB); at M = 513 (~16 MB matrix) they don't → every iteration streams from main memory → speed halves. **SpMV-based solvers are memory-bandwidth-bound** (week 2 caches → week 10 roofline).

### Week 4 — step 6 results: parallelism vs. preconditioner (2026-10-20)

Setup: opt build, M = 513, CG, `-ksp_rtol 1e-8 -ksp_norm_type unpreconditioned -random_rhs`, `-n 1, 2, 4` on my laptop.
- `-pc_type ilu` works only on 1 process (ILU is sequential). In parallel PETSc uses **block Jacobi** (`-pc_type bjacobi`, default sub-PC ILU(0)): ILU on each rank's own block, couplings between ranks ignored. On 1 process bjacobi = ILU.

| | n=1 | n=2 | n=4 |
|---|---|---|---|
| bjacobi its | 430 | 574 | 498 |
| gamg its | 15 | 15 | 15 |
| bjacobi time (s) | 1.653 | 1.647 | 0.904 |
| gamg time (s) | 0.647 | 0.558 | 0.328 |

- **GAMG**: iterations independent of p ✓ (coarse levels are built across all ranks — no information thrown away).
- **bjacobi**: 1 → 2 up 33% ✓ (cut couplings). 2 → 4 **down** (surprise). Possible reasons: (1) competing effects — more cut couplings (worse) but smaller blocks where ILU(0) is more accurate (better): strips 513×257 vs squares 257×257; (2) the random b is **not the same** for different p (see TODO).
- **Speedup** S(p) = T(1)/T(p): at p = 4 only **1.8× (bjacobi), 2.0× (gamg)**. Total Mflop/s (gamg) 1709 → 2353 → 3682 (2.2×). bjacobi at p = 2: no speedup — extra iterations eat the gain (per iteration it was 1.34× faster).
- **Why not 4×**: CG is memory-bandwidth-bound. Logically each MPI process has private memory; **physically** all processes on one laptop/node share the same RAM and memory bus → shared bandwidth saturates. Across cluster **nodes**, each node adds its own bandwidth → scaling continues further (until communication dominates).
- Interview: "Memory-bound codes stop scaling within a node once shared memory bandwidth is saturated. Adding cores doesn't add bandwidth; adding nodes does."

**Halo exchange visible in `-log_view`** (columns Mess = number of messages, AvgLen = bytes per message), bjacobi `KSPSolve`:
- p = 2 (strips 513×257): AvgLen 4.1e3 B = one grid row of 513 doubles × 8 B ✓; Mess 1.1e3 ≈ 574 its × 2 ranks × 1 neighbor ✓.
- p = 4 (squares 257×257): AvgLen 2.1e3 B = 257 doubles × 8 B ✓; Mess 4.0e3 ≈ 498 its × 4 ranks × 2 neighbors ✓.
→ One halo message per neighbor per MatMult; message length = block edge (strips vs squares from step 2.1).

TODO before CRC scaling runs:
- [ ] **Layout-independent random b**: `VecSetRandom` fills each rank's local part separately, so b changes with the number of processes → not exactly the same problem. Generate b from the global grid index (i, j) instead (e.g. a deterministic hash or a fixed smooth-but-rich function) so every p solves the identical system.

### Week 4 — step 7: benchmark script (2026-10-21)

`scripts/run_week4.sh` runs all week 4 cases and prints CSV; run with
`./scripts/run_week4.sh | tee results/week4.csv` (`tee` = write to file **and** screen; overwrites the file).
Columns: `ksp,pc,np,M,rep,its,reason,t_setup,t_solve`. 26 cases × 3 repeats = 78 runs.

Bash lessons:
- `set -euo pipefail` → stop on errors / unset variables / failed pipe parts (like `PetscCall`).
- `out=$(command)` runs the program **once** and stores all output; then parse it several times.
- Parsing: `grep -o 'its= [0-9]*' | awk '{print $2}'` (only the match, then 2nd word); `awk '/^KSPSolve /{print $4}'` (4th column of the line starting with KSPSolve).
- Functions: `run_case() { local ksp=$1 pc=$2 ...; }` — **no types, empty ()**; arguments are positional `$1, $2, …`; `local` = only inside the function. Call: `run_case cg $pc 1 $M`.
- `pc` is the text "pc", **`$pc` is the variable's value**. Missing `$` → PETSc got `-M M` → "Input string M has no integer value". PETSc's **option table** in the error shows exactly what the program received.
- `chmod +x` once to make a script executable. A bash function lives in the script, not in a .cpp file.
- New folders/files need `git add` (`-a` only covers already-tracked files).

Timing noise:
- Same case (GAMG, 1 proc, M = 513) measured in both experiments: setup 0.390 vs 0.556 s → single timings are unreliable.
- Fix: repeat each case (`rep` 1–3) and use the **minimum** time per case (least disturbed run).
- Repeats back-to-back agree within ~2%, but experiment 3 (run minutes later) was 10–25% slower → likely **thermal throttling** of the 15 W laptop CPU (hypothesis). Compare only runs made under the same conditions; speedups within experiment 3 are still fair.
- Laptop timings are indicative only → real scaling study on CRC (dedicated, cooled nodes).

### Week 4 — step 8: plots (2026-10-21)

`scripts/plot_week4.py` reads `results/week4.csv` → `results/iterations.png`, `results/time.png`.
- Install: `sudo apt install python3-pandas` (plain `pip install` is blocked on Ubuntu 24.04 — "externally managed environment").
- pandas: `pd.read_csv`; column arithmetic is element-wise (`df["t_total"] = df["t_setup"] + df["t_solve"]`); filter rows with `df[(cond1) & (cond2)]` (parentheses required); `groupby([...]).agg(t_total=("t_total","min"))` = one row per case, best of the repeats. `assert` that every run converged.
- 26 cases → 25 rows after groupby (GAMG 1 proc M = 513 appears in both experiments; bjacobi np=1 stays separate because its name differs from ilu).
- matplotlib: `matplotlib.use("Agg")` (write files, no window); `ax.set_xscale("log")`; log-log turns power laws into straight lines → read scaling from the slope. Dashed reference lines (∝ M, ∝ N, ∝ N^1.5), shifted down so they don't hide data. Fixed colors + different marker shapes per method (readable without color).
- Viewing from WSL: `cp results/time.png /mnt/c/Users/hdong2/Desktop/` (or `code results/time.png`).

What the plots show:
- **iterations.png** (x = M): none/jacobi/ilu parallel to slope 1 (∝ M ∝ 1/h, from κ ∝ 1/h²); ilu a constant factor lower; gamg almost flat (11 → 17).
- **time.png** (x = N = M²): crossover near N ≈ 17 000 (M = 129) — gamg slowest at the smallest size (setup), fastest beyond; at N ≈ 10⁶: 3 s vs 35 s. gamg ∥ ∝ N (optimal); none/jacobi/ilu ∥ ∝ N^1.5, **steeper** between M = 257 and 513 (cache kink: working set leaves L3). jacobi ≈ none in time (fewer iterations, slightly costlier ones).
