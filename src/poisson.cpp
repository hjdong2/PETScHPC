
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscvec.h>

int main (int argc, char** argv)
{
    DM da;
    Mat A;
    PetscInt xs, ys, xm, ym;
    PetscScalar v[5];
    PetscInt i, j, M=5, N=5, num;
    MatStencil row, col[5];
    PetscReal h;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
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
    h = 1.0 / (PetscReal)(M - 1);  // assume M == N now

    for (j = ys; j < ys + ym; ++j) {
        for (i = xs; i < xs + xm; ++i) {
            row.i = i;
            row.j = j;

            if (i == 0 || j == 0 || i == M - 1 || j == N - 1) {
                PetscScalar one = 1.0; 
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

    PetscBool isSym;
    MatIsSymmetric(A, 1e-12, &isSym);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "is symmetic %d \n", (int)isSym));

    PetscCall(MatDestroy(&A));
    PetscCall(DMDestroy(&da));

    PetscCall(PetscFinalize());

    return 0;
}
