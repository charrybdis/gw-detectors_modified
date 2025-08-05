#!/bin/bash -l
#PBS -l nodes=1:ppn=64
#PBS -l walltime=24:00:00
#PBS -r n
#PBS -j oe
#PBS -q starq

module load python/3.10.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
# filepath, frequency resolution, frequency factor, standard deviation, true coord resolution, psi/phi resolution, strain 
# resolution, workers, strain keys
python ./only_CE.py /fs/lustre/cita/jewelcao/results/only_CE/1_1e-3 200 1 1e-3 8 16 64 64 hb