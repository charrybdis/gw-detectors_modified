from gwdetectors import *
from Functions import *
import numpy as np

from mpi4py import MPI # this needs to be imported only where torque is installed or it gives annoying warnings 
from mpi4py.futures import MPIPoolExecutor

def produce_optimization_params(network, produce_freqs_signal_params, true_params, strain_keys, num): 
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
    numpts, spread, a, A, c, dt, p = produce_freqs_signal_params
    freqs, ast_signal = produce_freqs_signal(numpts, spread, a, A, c, dt, p)
    
    # true data information
    az_true, po_true, psi_true, geocent, coord, true_keys = true_params
    true_modes = dict.fromkeys(true_keys, ast_signal)
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
    current_dict = {}
    
    return info, true_params, current_dict, true_snr, optimization_variables, Coords_flat


def main_mpi(workers, num, Coords_flat, *args):
    """
    Runs one_variable_mp over Coords_flat with mpi4py. 
    Needs to be called in main with if __name__ =='__main__'!!
    Note that this will only work on the cluster since Torque is not installed elsewhere. 
    
    Parameters:
    workers --- max number of workers, set to None for automatic assignment
    num --- number of grid points in original azimuth, pole arrays
    Coords_flat --- list of coordinates in (azimuth, pole) format. Needs to be 1D.
    *args --- *args for one_variable_mp
    """
    
    one_variable = one_variable_mp(*args)

    with MPIPoolExecutor(max_workers=workers) as executor:
        results = executor.map(one_variable, Coords_flat)
    list_results = np.array(list(results))
    
    return list_results

