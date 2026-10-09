static char help[] = " hello code ";

#include <petscsys.h>

int main (int argc, char **argv)
{

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    
    PetscMPIInt rank, size;
    PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Running on %d processes\n", int(size)));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Rank: %d, Size: %d\n", (int)rank, (int)size));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

    PetscCall(PetscFinalize());
    return 0;
}
