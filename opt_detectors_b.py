from gwdetectors import *
from Functions import *
import numpy as np
import pandas as pd
import pickle
from time import perf_counter
import sys
import os

"""
loops over values of "true" sky coordinates for the injected data. 
optimizes over sky coordinates, polarization angle, psi, t0, and phi of comparison strain data.
returns overall best match, and corresponding max filter separately for each detector.
"""

#-------------------------------------------------------------------------------------------------
## define variables here, modify this section only

filepath = sys.argv[1]

# define network
K_instantiator, K_loc, K_arms = DETECTOR_ORIENTATIONS['K']
Kagra = K_instantiator('Kagra', PSDS[KNOWN_PSDS[0]], K_loc, K_arms, long_wavelength_approximation=False)
Perth = ECEF_CITIES['Perth']
CE_loc = Perth.to_lightseconds(Perth.location)
CE_arms = DETECTORS['CE@L_ce-design'].arms
CE_psd = DETECTORS['CE@L_ce-design'].psd
CE = TwoArmDetector('CE', CE_psd, CE_loc, CE_arms, long_wavelength_approximation=False)
network = Network(DETECTORS['H_aligo-design'], DETECTORS['L_aligo-design'], 
                  DETECTORS['V_advirgo-design'], Kagra, CE)
fsr = 1 / (2 * np.sum(DETECTORS['CE@L_ce-design'].arms[0]**2)**0.5)

# signal variables
freq_res = int(sys.argv[2])
spread = 4
a = fsr / float(sys.argv[3])
A = 1e-23
c = float(sys.argv[4])
dt = 0
p = 0

# true parameter variables
geocent = 0
coord = 'geographic'
true_keys = ['hb']
true_psi = 0
true_res = int(sys.argv[5])

# scipy brute variables
opt_func = filter_3 # remember if you change this you need to change variables and ranges
range_res = int(sys.argv[6])
ranges = ((0, 2*np.pi, 2*np.pi/range_res), (dt-0.04, dt+0.04, 0.08/8), (0, 2*np.pi, 2*np.pi/range_res))
npts = 20
variables = None

# azimuth, pole resolution
num = int(sys.argv[7])
workers = int(sys.argv[8]) # max number of worker processes

# strain modes
strain_keys = sys.argv[9:]

#---------------------------------------------------------------------------------------------------
## collecting variables, creating true sky positions

true_azimuths, true_poles = az_po_meshgrid(true_res, coord, truncate=True)

True_Az, True_Po = np.meshgrid(true_azimuths, true_poles, indexing='ij')
True_Psi = np.full(np.shape(True_Az.flatten()), true_psi)
missing_coords = [(0, 0, 0), (0, np.pi, 0)]
true_coords = list(zip(True_Az.flatten(), True_Po.flatten(), True_Psi)) + missing_coords 

produce_freqs_signal_params = [freq_res, spread, a, A, c, dt, p]
brute_params = [opt_func, ranges, npts]

#---------------------------------------------------------------------------------------------------
## dictionaries to store result info

if os.path.exists(f'{filepath}'):
    pass
else: 
    os.mkdir(f'{filepath}')

info = {'true_mode':true_keys, 'strain_mode':strain_keys,
        'a': a, 'A': A, 'c': c, 'dt':dt, 'phi':p, 
        'geocent':geocent, 'coord':coord, 'network':network, 
        'freq_res':freq_res, 'true_res':true_res, 'psi/phi_res':range_res, 'strain_res':num}

with open(f"{filepath}/info.pickle","wb") as f:
    pickle.dump(info, f)

#---------------------------------------------------------------------------------------------------
## run optimization

if __name__ == '__main__':
    iteration_start = perf_counter()
    for i, signal_coord in enumerate(true_coords): 
        i_start = perf_counter()
        
        run_results = true_coords_cf_detectors(signal_coord, true_psi, geocent, coord, true_keys,
                                               network, produce_freqs_signal_params, strain_keys, 
                                               num, brute_params, 
                                               variables=variables, 
                                               workers=workers, 
                                               finish=True)
        
        with open(f"{filepath}/results_{i}.pickle", "wb") as f:
            pickle.dump(run_results, f)
        
        i_end = perf_counter()
        elapsed_time = i_end - i_start
        print(f'Finished iteration {i} in {elapsed_time} seconds')
    
    iteration_end = perf_counter()
    print(f'Total time taken: {iteration_end - iteration_start} seconds')