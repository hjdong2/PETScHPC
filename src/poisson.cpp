
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscvec.h>

int main (int argc, char** argv)
{
    DM da;
    Mat A;
    PetscInt xs, ys, xm, ym;
    PetscScalar h, v[5];
    PetscInt i, j, M=8, N=8, num, numi, numj;
    MatStencil row, col[5]; 

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
    h = 1.0 / (PetscReal)(M - 1);

    for (j = ys; j < ys + ym; ++j) {
        for (i = xs; i < xs + xm; ++i) {
            row.i = i;
            row.j = j;

            if (i == 0 || j == 0 || i == M - 1 || j == N - 1) {
                num = 0;
                numi = 0;
                numj = 0;

                if (j != 0) {
                    v[num] = -1.0;
                    col[num].i = i;
		    col[num].j = j - 1;
                    num++;
                    numj++;
		}

                if (i != 0) {
                    v[num] = -1.0;
                    col[num].i = i - 1;
                    col[num].j = j;
                    num++;
                    numi++;
		}

                if (i != M - 1) {
                    v[num] = -1.0;
                    col[num].i = i + 1;
                    col[num].j = j;
                    num++;
                    numi++;
		}

                if (j != N - 1) {
                    v[num] = -1.0;
                    col[num].i = i;
                    col[num].j = j + 1;
                    num++;
                    numj++;
		}

                v[num] = (PetscReal)numj * 1.0 + (PetscReal)numi * 1.0;
                col[num].i = i;
                col[num].j = j;
                num++;
                PetscCall(MatSetValuesStencil(A, 1, &row, num, col, v, INSERT_VALUES));
	    } else {
                v[0] = -1.0;
                col[0].i = i;
                col[0].j = j - 1;

                v[1] = -1.0;
                col[1].i = i - 1;
                col[1].j = j;

                v[2] = 2.0 * (1.0 + 1.0);
                col[2].i = i;
                col[2].j = j;

                v[3] = -1.0;
                col[3].i = i + 1;
                col[3].j = j;

                v[4] = -1.0;
                col[4].i = i;
                col[4].j = j + 1;
                PetscCall(MatSetValuesStencil(A, 1, &row, 5, col, v, INSERT_VALUES));
	    }
	}
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(MatDestroy(&A));
    PetscCall(DMDestroy(&da));

    PetscCall(PetscFinalize());

    return 0;
}
