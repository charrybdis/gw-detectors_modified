from gwdetectors import *
from Functions import *
from cluster import *
import numpy as np
import pandas as pd

"""
tries combinations of detector locations for a given arm angle. 
optimizes over sky coordinates, polarization angle, psi, t0, and phi of comparison strain data,
returns best match, parameters of best match
"""

#-------------------------------------------------------------------------------------------------
## define variables here, modify this section only

# define network
H_instantiator, H_loc, H_arms = DETECTOR_ORIENTATIONS['H']
fsr = 1 / (2 * np.sum(DETECTORS['H_aligo-design'].arms[0]**2)**0.5)
det_res = 4
det_arms = H_arms # this can be the same or different for all detectors

# signal variables
numpts = 120
spread = 4
a = fsr 
A = 1e-23
c = 0.5
dt = 0
p = 0

# true parameter variables
true_az = 0
true_po = 0
true_psi = 0 
geocent = 0
coord = 'geographic'
true_keys = ['hp']
true_res = 4
true_psi = 0

# scipy brute variables
opt_func = filter_2a
ranges = ((0, np.pi, np.pi/90), (0, 2*np.pi, 2*np.pi/180)) #(dt-0.01, dt+0.01, 0.01) if including time
npts = 20
finish_func = None

# strain modes
strain_keys = ['hvx']

# azimuth, pole resolution
num = 40

#---------------------------------------------------------------------------------------------------
## collecting variables, creating detector positions

latitudes = np.linspace(-90, 90, det_res)
longitudes = np.linspace(-180, 180, det_res * 2)
Lats, Lons = np.meshgrid(latitudes, longitudes, indexing='ij')
geo_Lats = Lats.flatten()
geo_Lons = Lons.flatten()

det_locs = np.array([(geodetic_to_ECEF(geo_Lats[i], geo_Lons[i], 0)) for i in range(len(geo_Lats))])

new_detectors = [H_instantiator(f'det{i}', 
                                PSDS[KNOWN_PSDS[0]], 
                                det_loc, 
                                det_arms, 
                                long_wavelength_approximation=False) for i, det_loc in enumerate(det_locs)]

K_instantiator, K_loc, K_arms = DETECTOR_ORIENTATIONS['K']
Kagra = K_instantiator('Kagra', PSDS[KNOWN_PSDS[0]], K_loc, K_arms, long_wavelength_approximation=False)
network_initial = Network(DETECTORS['H_aligo-design'], DETECTORS['L_aligo-design'], 
                  DETECTORS['V_advirgo-design'], DETECTORS['CE@L_ce-design'], Kagra)

produce_freqs_signal_params = [numpts, spread, a, A, c, dt, p]
brute_params = [opt_func, ranges, npts, finish_func]
true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]

#---------------------------------------------------------------------------------------------------
## dictionaries to store result info

info = {'a': a, 'A': A, 'c': c, 'dt':dt, 'phi':p, 
        'azim':true_az, 'pole':true_po, 'psi': true_psi, 
        'modes':true_keys[0], 'geocent':geocent, 'coord':coord}

df_info = pd.DataFrame.from_dict([info])
current_dict={}

#---------------------------------------------------------------------------------------------------
## run optimization
if __name__ == '__main__':
    match, best_detector = detector_iterate(new_detectors, network_initial, produce_freqs_signal_params, 
                                            true_params, strain_keys, num, brute_params)
    
    current_dict['match'] = match
    current_dict['best_detector'] = best_detector

#---------------------------------------------------------------------------------------------------
## save results to file
df_results = pd.DataFrame.from_dict(current_dict)

df_results.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/{true_params[-1]}_true.csv')
df_info.to_csv(f'/fs/lustre/cita/jewelcao/gw-detectors/results/info.csv')