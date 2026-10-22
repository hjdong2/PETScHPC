#!/bin/bash
set -euo pipefail

MPIEXEC=$HOME/petsc/arch-linux-c-opt/bin/mpiexec
EXE=./build-opt/poisson

run_case() {
    local ksp=$1 pc=$2 np=$3 M=$4 rep=$5
    out=$($MPIEXEC -n $np $EXE -M $M -ksp_type $ksp -pc_type $pc \
        -ksp_rtol 1e-8 -ksp_norm_type unpreconditioned -random_rhs -log_view)

    its=$(echo "$out" | grep -o 'its= [0-9]*' | awk '{print $2}')
    reason=$(echo "$out" | grep -o 'reason= [A-Z_]*' | awk '{print $2}')
    t_setup=$(echo "$out" | awk '/^PCSetUp /{print $4}')
    t_solve=$(echo "$out" | awk '/^KSPSolve /{print $4}')

    echo "$ksp,$pc,$np,$M,$rep,$its,$reason,$t_setup,$t_solve"
}

echo "ksp,pc,np,M,rep,its,reason,t_setup,t_solve"

for pc in none jacobi ilu gamg; do
    for M in 65 129 257 513 1025; do
        for rep in 1 2 3; do
            run_case cg $pc 1 $M $rep
        done
    done
done

for pc in bjacobi gamg; do
    for np in 1 2 4; do
        for rep in 1 2 3; do
            run_case cg $pc $np 513 $rep
        done
    done
done
