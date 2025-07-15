#!/bin/bash -l
#PBS -l nodes=2:ppn=16
#PBS -l walltime=48:00:00
#PBS -r n
#PBS -j oe
#PBS -q hpq

module load python/3.10.2 
module load openmpi/4.1.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
python ./optimization.py 