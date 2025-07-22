from gwdetectors import *
import Functions
from cluster import *
import numpy as np
import pandas as pd

## define variables here

# define network
network = Network(DETECTORS['H_aligo-design'], DETECTORS['L_aligo-design'], 
                  DETECTORS['V_advirgo-design'])

# signal variables
numpts = 120
spread = 4
a = 100 
A = 1e-23
c = 1
dt = 0
p = 0

# true parameter variables
geocent = 0
true_az = 0
true_po = 0
true_psi = 0
coord = 'geographic'
true_keys = ['hp']

# scipy brute variables
opt_func = Functions.filter_3
ranges = ((0, np.pi, np.pi/90), (-0.01, +0.01, 0.01), (0, 2*np.pi, 2*np.pi/180))
npts = 20
finish_func = None

# strain modes
strain_keys = ['hvx']

# azimuth, pole resolution
num = 45

#---------------------------------------------------------------------------------------------------

## dictionaries to store result info

info = {'a': a, 'A': A, 'c': c, 'dt':dt, 'network':network}
true_params = {'pole': po_true, 'azim':az_true,
               'psi': psi_true, 't0':dt, 'phi'=p,
               'geocent':geocent, 'modes':true_keys
              }
current_dict={}
#---------------------------------------------------------------------------------------------------

## collect variables
produce_freqs_signal_params = [numpts, spread, a, A, c, dt, p]
true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]
brute_params = [opt_func, ranges, npts, finish_func]

## run optimization
if __name__ == '__main__':
    true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params, true_params, strain_keys, num)
    list_results = main_mpi(None, num, Coords_flat, *brute_params, optimization_variables)

filter_grid = np.reshape(list_results, (num-1, num-1))

## save results, parameters
max_filter = np.max(filter_grid)
rho_match = max_filter / true_snr

max_skyindex = np.where(list_results == np.max(list_results))
max_sky_coords = [Coords_flat[i] for i in max_skyindex]
max_az, max_po = max_sky_coords[0]

max_vars = Functions.brute_max(Functions.filter_2a, ranges_slice, npts, finish_fun, *optimization_variables, 
                               max_az, max_po)[0]
run_results = {'filter':max_filter, 'match':rho_match, 'params':[max_az, max_po, *max_vars]}

current_dict[strain_keys[0]] = run_results

np.savetxt(f"/fs/lustre/cita/jewelcao/gw-detectors/results/{true_params[-1]}_{strain_keys[0]}.txt", filter_grid)

df_info = pd.DataFrame.from_dict(info)
df_true_params = pd.DataFrame.from_dict(true_params)
df_results = pd.DataFrame.from_dict(current_dict)
df_results.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/{true_params[-1]}_true.csv')
df_info.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/info.csv')
df_true_params.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/params.csv')