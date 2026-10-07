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
| 2 | Memory hierarchy, caches, cache lines, locality | Eijkhout (single-processor chapter); Drepper (CPU caches, what programmers can do) | [ ] |
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

Notes:

