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
optimizes over sky coordinates, polarization angle, psi, t0, and phi of comparison strain data,
returns best match, parameters of best match
"""

#-------------------------------------------------------------------------------------------------
## define variables here, modify this section only

filepath = sys.argv[1]

# define network
CE_loc = np.array([-1000, 4000, 4000]) * 1000 / 299792458
CE_arms = DETECTORS['CE@H_ce-design'].arms
CE_psd = DETECTORS['CE@H_ce-design'].psd
CE = TwoArmDetector('CE', CE_psd, CE_loc, CE_arms, long_wavelength_approximation=False)
fsr = 1 / (2 * np.sum(DETECTORS['CE@L_ce-design'].arms[0]**2)**0.5)

# signal variables
freq_res = int(sys.argv[2])
spread = 4
a = fsr 
A = 1e-23
c = 1e-2
dt = 0
p = 0

# true parameter variables
geocent = 0
coord = 'geographic'
true_keys = ['hp']
true_psi = 0
true_res = int(sys.argv[3])

# scipy brute variables
opt_func = filter_3_det # remember if you change this you need to change variables and ranges
range_res = int(sys.argv[4])
ranges = ((0, 2*np.pi, 2*np.pi/range_res), (0, 2*np.pi, 2*np.pi/range_res), (dt-0.04, dt+0.04, 0.08/4))
npts = 20
variables = None

# strain modes
strain_keys = sys.argv[7:]

# azimuth, pole resolution
num = int(sys.argv[5])
workers = int(sys.argv[6]) # max number of worker processes

#---------------------------------------------------------------------------------------------------
## collecting variables, creating true sky positions

true_azimuths, true_poles = az_po_meshgrid(true_res, coord)

True_Az, True_Po = np.meshgrid(true_azimuths, true_poles, indexing='ij')
True_Psi = np.full(np.shape(True_Az.flatten()), true_psi)
true_coords = list(zip(True_Az.flatten(), True_Po.flatten(), True_Psi)) 

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
        'geocent':geocent, 'coord':coord, 'detector':CE, 
        'freq_res':freq_res, 'true_res':true_res, 'psi/phi_res':range_res, 'strain_res':num}

with open(f"{filepath}/info.pickle","wb") as f:
    pickle.dump(info, f)

#---------------------------------------------------------------------------------------------------
## run optimization

if __name__ == '__main__':
    iteration_start = perf_counter()
    for i, signal_coord in enumerate(true_coords): 
        i_start = perf_counter()
        
        run_results = true_coords_cf(signal_coord, true_psi, geocent, coord, true_keys,
                                     CE, produce_freqs_signal_params, strain_keys, 
                                     num, brute_params, 
                                     variables=variables, 
                                     workers=workers, 
                                     finish=True, 
                                     full_output=True)
        
        with open(f"{filepath}/results_{i}.pickle", "wb") as f:
            pickle.dump(run_results, f)
        
        i_end = perf_counter()
        elapsed_time = i_end - i_start
        print(f'Finished iteration {i} in {elapsed_time} seconds')
    
    iteration_end = perf_counter()
    print(f'Total time taken: {iteration_end - iteration_start} seconds')