
#include <petsc.h>
static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
    -view_exact_sol    : write exact solution vector to stdout\n\
    -m <mesh_x>        : number of mesh points in x-direction\n\
    -n <mesh_y>        : number of mesh points in y-direction\n\n";

int main(int argc, char** args)
{
    Vec          x, b, u;
    Mat          A;
    KSP          ksp;
    PetscReal    norm;
    PetscInt     i, j, Ii, J, Istart, Iend, m = 8, n = 7, its;
    PetscBool    flg;
    PetscScalar  v;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &args, NULL, help));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));


    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
    PetscCall(MatSetFormOptions(A));
    PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL));
    PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
    PetscCall(MatSeqSBAIJSetPreallocation(A, 1, 5, NULL));
    PetscCall(MatMPISBAIJSetPreallocation(A, 1, 5, NULL, 5, NULL));
    PetscCall(MatMPISELLSetPreallocation(A, 5, NULL, 5, NULL));
    PetscCall(MatSeqSELLSetPreallocation(A, 5, NULL));

    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

    for (Ii = Istart; Ii < Iend; ++Ii) {
        v = -1.0;
	i = Ii / n;
	j = Ii - i * n;
	if (i > 0) {
            J = Ii - n;
	    PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
	}

        if (i < m - 1) {
            J = Ii + n;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
	}

        if (j > 0) {
            J = Ii - 1;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
	}

        if (j < n - 1) {
            J = Ii + 1;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
	}

        v = 4.0;
        PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
    
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));

    PetscCall(VecCreate(PTETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PTESC_DECIDE, m * n));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecDuplicate(u, &b));
    PetscCall(VecDuplicate(b, &x));

    PetscCall(VecSet(u, 1.0));
    PetscCall(MatMult(A, u, b));

    flg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_exact_sol", &flg, NULL));
    if (flg) PetscCall(VecView(u, PETSC_VIEW_STDOUT_WORLD));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

    PetscCall(KSPSetOperators(ksp, A, A));

    PetscCall(KSPSetTolerances(ksp, 1.e-2 / ((m + 1) * (n + 1)), 1.e-50, PETSC_CURRENT, PETSC_CURRENT));

    PetscCall(KSPSetFromOptions(ksp));

    PetscCall(KSPSolve(ksp, b, x));

    PetscCall(VecAXPY(x, -1.0, u));
    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCall(KSPGetIterationNumber(ksp, &its));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %" PetscInt_FMT "\n", (double)norm, its));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));

    PetscCall(PetscFinalize());

    return 0;
}
