#!/bin/bash -l
#PBS -l nodes=1:ppn=128
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -J 1-5
#PBS -q starq

module load python/3.10.2

# go to your working directory containing the batch script, code and data
cd /fs/lustre/cita/jewelcao/gw-detectors
# filepath, frequency resolution, frequency factor, standard deviation, true coord resolution, psi/phi resolution, strain 
# resolution, workers, strain keys
frequencies=(1 "0.4" "0.8" "1.2" "1.6" "2.0") # since array starts at 1, excludes hp
current_frequency=${frequencies[$PBS_ARRAY_INDEX]}
export NUMEXPR_MAX_THREADS=128
python ./optimization_cf_detectors.py \
/fs/lustre/cita/jewelcao/results/hp_true/hb/${current_frequency}_1e-3 200 ${current_frequency} 5e-3 8 16 64 128 hb