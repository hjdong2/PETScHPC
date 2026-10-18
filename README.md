# Parallel 2D Poisson solver with PETSc 

PETScHPC solves a Poisson equation $-\Delta u = f$ on $\Omega: (0, 1)^2$ with boundary condition $u = 0$ on $\partial \Omega$
by 5-point stencil central difference method.
It is a C++17 code that uses PETSc for the distributed grid (DMDA), sparse matrix assembly and Krylov solvers (KSP).
The solver is verified with a manufactured solution and the error drops by a 
factor of 4.00 each time mesh size $h$ is halved $(O(h^2))$.
The solver runs in parallel with MPI plus PETSc DMDA domain decomposition, and the 
results show identical error ($5.02e-5$ at $129 \times 129$) on 1 and 4 processes.
