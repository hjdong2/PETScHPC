#!/bin/bash
#$ -N poisson_test          # job name (shows in qstat)
#$ -q hpc@d32cepyc097       # your group's queue
#$ -pe smp 64         # ask for 64 cores = 1 full node
#$ -cwd                     # run in the directory where you called qsub
#$ -j y                     # stdout and stderr in one file
#$ -o poisson_test.$JOB_ID.out   # output file name; $JOB_ID is filled in by SGE

module load petsc/3.24.1

echo "Running on $(hostname) with $NSLOTS slots"
mpiexec -n 4 ./build/poisson -M 513 -ksp_type cg -pc_type gamg \
        -ksp_rtol 1e-8 -ksp_norm_type unpreconditioned -random_rhs -log_view
