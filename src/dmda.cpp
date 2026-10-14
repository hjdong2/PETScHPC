
#include <petscdmda.h>

int main(int argc, char** argv)
{
    DM da;
    PetscInt Nx = 8, Ny = 8;
    PetscInt xs, ys, xm, ym;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(DMDACreate2d(
        PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE,
        DM_BOUNDARY_NONE,
        DMDA_STENCIL_STAR,
        Nx, Ny,
        PETSC_DECIDE, PETSC_DECIDE,
        1, 1,
        NULL, NULL,
        &da
    ));

    PetscCall(DMSetUp(da));

    PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

    PetscMPIInt rank, size;
    PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Rank: %d, Size: %d\n", (int)rank, (int)size));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "xs: %" PetscInt_FMT "\n", xs));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "ys: %" PetscInt_FMT "\n", ys));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "xm: %" PetscInt_FMT "\n", xm));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "ym: %" PetscInt_FMT "\n", ym));

    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
   
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());

    return 0;
}
