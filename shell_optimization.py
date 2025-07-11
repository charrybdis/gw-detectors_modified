from gwdetectors import *
import Functions
import numpy as np
import pandas as pd

# define variables here
network = Network(DETECTORS['H_aligo-design'], DETECTORS['L_aligo-design'], 
                  DETECTORS['V_advirgo-design'])
produce_freqs_signal_params = [120, 4, 100, 1e-23, 1]
true_params = [0,0,0,0,'geographic', ['hp']]
brute_params = [Functions.filter3, ((0, np.pi, np.pi/45), (dt-0.05, dt+0.05, 0.1/2), (0, 2*np.pi, 2*np.pi/90)), 20, None]
strain_keys = ['hx']
num = 45

def produce_optimization_params(network, produce_freqs_signal_params, true_params, brute_params, strain_keys, num): 
    """
    Creates all the information needed to optimize the filter response over psi, t0, phi and calculate it over a grid.
    
    Parameters: 
    network --- instance of Network containing Detectors
    produce_freqs_signal_params --- parameters for produce_freqs_signal function in Functions
    true_params --- parameters for "true" data; azimuth, pole, psi, geocent_time, {hp:signal, hx:signal,....} 
    brute_params --- parameters for scipy brute function
    strain_keys --- ['hp', 'hx',...]
    num --- number of grid points for sky 
    """
    
    # signal information
    numpts, spread, a, A, c, dt = produce_freqs_signal_params
    freqs, ast_signal = Functions.produce_freqs_signal(numpts, spread, a, A, c, dt)
    
    # true data information
    az_true, po_true, psi_true, geocent, coord, true_keys = true_params
    true_modes = dict.fromkeys(keys, ast_signal)
    data = network.project(freqs, geocent, az_true, po_true, psi_true, coord=coord, **true_modes)
    true_snr = network.snr(freqs, geocent, az_true, po_true, psi_true, coord=coord, **true_modes)
    
    # pack optimization variables
    optimization_variables = [a, A, c, network, freqs, geocent, data, coord, strain_keys]
    
    # generate sky coordinate pairs
    azimuths = np.linspace(-np.pi, np.pi, num)
    poles = np.flip(np.linspace(0, np.pi, num))

    Azimuths, Poles = np.meshgrid(azimuths[1:], poles[1:], indexing='ij')
    Azimuths_flat = Azimuths.flatten()
    Poles_flat = Poles.flatten()
    Coords_flat = list(zip(Azimuths_flat, Poles_flat))

    # Create dictionary with true parameters to store results
    info = {'a': a, 'A': A, 'c': c, 'dt':dt, 'network':network}
    true_params = {'pole': po_true, 'azim':az_true,
                   'psi': psi_true, 't0':dt,
                   'geocent':geocent, 'modes':true_keys
                  }
    current_dict = {'info':info, 'true_params':true_params}
    
    return true_snr, current_dict, brute_params, optimization_variables, Coords_flat


    if __name__ == '__main__':
        true_snr, current_dict, brute_params, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params, true_params, brute_params, strain_keys, num)
        filter_grid = Functions.main_mpi(None, num, Coords_flat, 
                                         *brute_params, optimization_variables)

        max_filter = np.max(filter_grid)
        rho_match = max_filter / true_snr
        run_results = {'filter':max_filter, 'match':rho_match}

        current_dict[keys[0]] = run_results

        np.savetxt(f"/fs/lustre/cita/jewelcao/results/{list(true_kwargs.keys())[0]}_{keys[0]}.txt", filter_grid)

        df = pd.DataFrame.from_dict(current_dict)
        df.to_csv(f'/fs/lustre/cita/jewelcao/results/{list(true_kwargs.keys())[0]}_true.csv')