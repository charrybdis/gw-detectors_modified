#!/bin/bash -l
#PBS -l nodes=1:ppn=64
#PBS -l walltime=12:00:00
#PBS -r n
#PBS -j oe
#PBS -q starq

module load python/3.10.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
python ./optimization_true_coords_cf.py