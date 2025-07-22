#!/bin/bash -l
#PBS -l nodes=1:ppn=32
#PBS -l walltime=12:00:00
#PBS -r n
#PBS -j oe
#PBS -q greenq

module load python/3.10.2 
module load openmpi/4.1.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
env MPI4PY_FUTURES_MAX_WORKERS=32 mpiexec -n 1 python ./optimization_loop.py 