#!/bin/bash -l
#PBS -l nodes=1:ppn=8
#PBS -l walltime=48:00:00
#PBS -r n
#PBS -j oe
#PBS -q workq

module load python/3.10.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
./shell_optimization.py 