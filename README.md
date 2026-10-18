
# Parallel 2D Poisson solver with PETSc

PETScHPC solves a poisson equation $-\Delat u = f$ on $\Omega: [0, 1]^2$ with boundary condition $u = 0$ on $\partial \Omega$
by 5-stencil central difference method.
It is a C++17 code that uses PETSc library to implement Poisson solve.
The solver is verified by manufactured solution and shows second-order accuracy.
The solver runs in parallel with MPI plus PETSc DMDA domain decomposition, and the 
results show no difference between 1 and 4 processes.
