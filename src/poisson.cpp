
#include <petscdmda.h>
#include <petscksp.h>

int main (int argc, char** argv)
{
    DM da;
    Mat A;
    Vec b, x, u;
    KSP ksp;
    PetscInt xs, ys, xm, ym;
    PetscScalar v[5];
    PetscInt i, j, M=5, N, num;
    MatStencil row, col[5];
    PetscReal h, err;
    PetscScalar **barr, **uarr;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
    N = M;

    auto onBoundary = [M, N](PetscInt i, PetscInt j) {
        return (i == 0 || j == 0 || i == M - 1 || j == N - 1);
    };

    PetscCall(DMDACreate2d(
        PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE,
        DM_BOUNDARY_NONE,
        DMDA_STENCIL_STAR,
        M, N,
        PETSC_DECIDE, PETSC_DECIDE,
        1, 1,
        NULL, NULL,
        &da
    ));

    PetscCall(DMSetUp(da));

    PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

    PetscCall(DMCreateMatrix(da, &A));
    PetscCall(DMCreateGlobalVector(da, &b));
    PetscCall(VecDuplicate(b, &x));
    PetscCall(VecDuplicate(b, &u));

    h = 1.0 / (PetscReal)(M - 1);  // assume M == N now

    for (j = ys; j < ys + ym; ++j) {
        for (i = xs; i < xs + xm; ++i) {
            row.i = i;
            row.j = j;

            if (onBoundary(i, j)) {
                PetscScalar one = 1.0;  // DBC 
                PetscCall(MatSetValuesStencil(A, 1, &row, 1, &row, &one, INSERT_VALUES));
            } else {
                num = 0;

                v[num] = 4.0 / (h * h);
                col[num].i = i;
                col[num].j = j;
                num++;

                if (j - 1 > 0) {
                    v[num] = -1.0 / (h * h);
                    col[num].i = i;
                    col[num].j = j - 1;
                    num++;
                }

                if (i - 1 > 0) {
                    v[num] = -1.0 / (h * h);
                    col[num].i = i - 1;
                    col[num].j = j;
                    num++;
                }

                if (i + 1 < M - 1) {
                    v[num] = -1.0 / (h * h);
                    col[num].i = i + 1;
                    col[num].j = j;
                    num++;
                }

                if (j + 1 < N - 1) {
                    v[num] = -1.0 / (h * h);
                    col[num].i = i;
                    col[num].j = j + 1;
                    num++;
                }

                PetscCall(MatSetValuesStencil(A, 1, &row, num, col, v, INSERT_VALUES));
            }
        }
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(DMDAVecGetArray(da, b, &barr));
    for (j = ys; j < ys + ym; ++j) {
        for (i = xs; i < xs + xm; ++i) {
            if (!onBoundary(i, j)) {
                barr[j][i] = 2 * PETSC_PI * PETSC_PI  * PetscSinReal(PETSC_PI * i * h) * PetscSinReal(PETSC_PI * j * h);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da, b, &barr));
    
    PetscCall(DMDAVecGetArray(da, u, &uarr));
    for (j = ys; j < ys + ym; ++j) {
        for (i = xs; i < xs + xm; ++i) {
             uarr[j][i] = PetscSinReal(PETSC_PI * i * h) * PetscSinReal(PETSC_PI * j * h);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, u, &uarr));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));

    PetscInt its;
    PetscCall(KSPGetIterationNumber(ksp, &its));

    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp, &reason));

    PetscCall(VecAXPY(x, -1.0, u));
    PetscCall(VecNorm(x, NORM_INFINITY, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "M= %" PetscInt_FMT ", h = %g, error= %g, its= %" PetscInt_FMT ", reason= %s\n", 
        M, (double)h, (double)err, its, KSPConvergedReasons[reason]));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&u));
    PetscCall(DMDestroy(&da));

    PetscCall(PetscFinalize());

    return 0;
}
