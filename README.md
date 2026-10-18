# Parallel 2D Poisson solver with PETSc 

PETScHPC solves a Poisson equation $-\Delta u = f$ on $\Omega: (0, 1)^2$ with boundary condition $u = 0$ on $\partial\Omega$
by 5-point stencil central difference method.
It is a C++17 code that uses PETSc for the distributed grid (DMDA), sparse matrix assembly and Krylov solvers (KSP).
The solver is verified with a manufactured solution and the error drops by a 
factor of 4.00 each time mesh size $h$ is halved $(O(h^2))$.
The solver runs in parallel with MPI plus PETSc DMDA domain decomposition, and the 
results show identical error ($5.02e-5$ at $129 \times 129$) on 1 and 4 processes.

## Problem and method

1. Problem and the manufactured solution
$-\Delta u = f \quad \text{ in } \Omega = (0, 1)^2, \qquad u = 0 \text{ on } \partial\Omega$
with $f = 2 \pi^2 \sin(\pi x) \sin(\pi y).$
The manufactured (exact) solution is $u = \sin(\pi x) \sin(\pi y)$,
the manufactured solution vanishes on the boundary, which fits the boundary condition exactly.

2. Discretization
- The mesh/grid: $M \times N$ points including the boundary with mesh size $h = 1/ (M - 1)$.
- 5-point stencil scheme
$-\frac{u_{i+1, j} - 2u_{i,j} + u_{i-1, j} + u_{i, j+1} - 2u_{i,j} + u_{i, j-1}}{h^2} = f_{i, j} $

3. Boundary condition treatment
- All points on the boundary, i.e., $i = 0$, $i = M - 1$, $j = 0$, $j = N - 1$, are given an identity row (1 on the diagonal, right-hand side 0), which enforces $u_{i, j} = 0$ (Dirichlet BC).
- For interior points that have a boundary point as a neighbor, the boundary neighbor's entry is left out of the row, so the row has fewer than 5 entries. The neighbor's value is known to be 0, so its term in the stencil is 0.
- The boundary columns are left out of the interior rows, so the matrix $A$ to solve the final linear system $Ax = b$ is SPD. This allows to use Conjugate Gradient (CG) method, `-ksp_type cg`, which is the natural Krylov solver for SPD systems.

4. Parallel implementation
- PETSc DMDA splits the mesh into blocks, one per MPI process, and each process assembles only the rows it owns.
- Rows and columns are set with mesh indices (`MatSetValuesStencil`).
- Solved with Krylov linear solver (KSP), with the solver and preconditioner chosen at the run time (`-ksp_type`, `-pc_type`).
