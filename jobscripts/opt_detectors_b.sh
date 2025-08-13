#!/bin/bash -l
#PBS -l nodes=1:ppn=128
#PBS -l walltime=6:00:00
#PBS -r n
#PBS -j oe
#PBS -q starq

module load python/3.10.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
# filepath, frequency resolution, frequency factor, standard deviation, true coord resolution, psi/phi resolution, strain 
# resolution, workers, strain keys
export NUMEXPR_MAX_THREADS=128
python ./opt_detectors_b.py /fs/lustre/cita/jewelcao/results/hb_true/hl/1_1e-3 200 1 1e-3 8 16 64 128 hl