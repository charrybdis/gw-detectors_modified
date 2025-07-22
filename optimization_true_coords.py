from gwdetectors import *
from Functions import *
from cluster import *
import numpy as np
import pandas as pd

"""
loops over values of "true" sky coordinates for the injected data. 
optimizes over sky coordinates, polarization angle, psi, t0, and phi of comparison strain data,
returns best match, parameters of best match
"""

#-------------------------------------------------------------------------------------------------
## define variables here, modify this section only

# define network
K_instantiator, K_loc, K_arms = DETECTOR_ORIENTATIONS['K']
Kagra = K_instantiator('Kagra', PSDS[KNOWN_PSDS[0]], K_loc, K_arms, long_wavelength_approximation=False)
network = Network(DETECTORS['H_aligo-design'], DETECTORS['L_aligo-design'], 
                  DETECTORS['V_advirgo-design'], DETECTORS['CE@L_ce-design'], Kagra)
fsr = 1 / (2 * np.sum(DETECTORS['H_aligo-design'].arms[0]**2)**0.5)

# signal variables
numpts = 120
spread = 4
a = fsr 
A = 1e-23
c = 1e-3
dt = 0
p = 0

# true parameter variables
geocent = 0
coord = 'geographic'
true_keys = ['hp']
true_res = 4
true_psi = 0

# scipy brute variables
opt_func = filter_2a # remember if you change this you need to go change optimization_variables
ranges = ((0, np.pi, np.pi/45), (0, 2*np.pi, 2*np.pi/90)) #(dt-0.01, dt+0.01, 0.01) if including time
npts = 20
finish_func = None

# strain modes
strain_keys = ['hb']

# azimuth, pole resolution
num = 32

#---------------------------------------------------------------------------------------------------
## collecting variables, creating true sky positions

true_azimuths, true_poles = az_po_meshgrid(true_res, coord)

True_Az, True_Po = np.meshgrid(true_azimuths, true_poles, indexing='ij')
True_Psi = np.full(np.shape(True_Az.flatten()), true_psi)
true_coords = list(zip(True_Az.flatten(), True_Po.flatten(), True_Psi)) 

produce_freqs_signal_params = [numpts, spread, a, A, c, dt, p]
brute_params = [opt_func, ranges, npts, finish_func]

#---------------------------------------------------------------------------------------------------
## dictionaries to store result info

info = {'a': a, 'A': A, 'c': c, 'dt':dt, 'phi':p, 'modes':true_keys[0], 
        'geocent':geocent, 'coord':coord, 'network':network}

df_info = pd.DataFrame.from_dict(info)
df_info.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/current_run/info.csv')

#---------------------------------------------------------------------------------------------------
## run optimization

if __name__ == '__main__':
    for i, signal_coord in enumerate(true_coords): 
        run_results = true_coords_(signal_coord, true_psi, geocent, coord, true_keys,
                                   network, produce_freqs_signal_params, strain_keys, num,
                                   brute_params, full_output=False)
        df_results = pd.DataFrame.from_dict(run_results)
        df_results.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/current_run/{"_".join(true_keys) + "-" + "_".join(strain_keys)}_{i}.csv')
        print(f'Finished iteration {i}')