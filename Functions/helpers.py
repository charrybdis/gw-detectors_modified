import numpy as np
import warnings
from .general_functions import * 
from .optimization_functions import *
from functools import partial

#----------------------------------------------------------------------------------------------
### multiprocessing over strain sky coordinates azimuth, pole
# maximizes over an inner layer (strain sky coordinates)

def one_variable_filter_mp(function, ranges, numpoints, optimization_variables, coordinates, finish=False): 
    """
    Helper for main_cf.
    Same as brute_max, but with azimuth, pole as a single coordinate tuple, and returns only optimization result, 
    not variables. 
    
    Parameters:
    function, ranges, numpoints, finish_func --- parameters in brute
    optimization_variables --- (List) variables for the optimization function
    """
    az, po = coordinates
    full_func = brute_max(function, ranges, numpoints, *optimization_variables, az, po, finish=finish)[1]
    
    return full_func


def one_variable_mp(*args, finish=False):
    """
    Helper for main_cf. 
    Returns function one_variable_filter_mp with just one argument (coordinates=azimuth, pole), for use in map.
    
    Parameters: 
    *args --- all the parameters for one_variable_filter_mp, EXCEPT coordinates
    """
    one_variable = partial(one_variable_filter_mp, *args, finish=finish)
    
    return one_variable

#----------------------------------------------------------------------------------------------
# more complicated functions with multiprocessing

def produce_optimization_params(detectors, produce_freqs_signal_params, true_params, strain_keys, num, variables=None): 
    """
    Helper for true_coords_cf. 
    Creates all the information needed to optimize the filter response over psi, phi, t0, and strain sky coordinates.

    Parameters: 
    detectors --- (Detector or Network) instance of Network or Detector class. 
    produce_freqs_signal_params --- (List) parameters for produce_freqs_signal function in Functions
    true_params --- (List) parameters for "true" data; azimuth, pole, psi, geocent_time, coord, {hp:signal, hx:signal,....} 
    strain_keys --- (List of strings) ['hp', 'hx',...]
    num --- (Int) number of grid points for sky
    variables --- (List) Defaults to None, corresponding to optimization over all 3 variables (psi, phi, t0). Otherwise, 
        should be a list containing the fixed values of the variables not being optimized over, i.e. if using filter_2a,
        variables should be [t0].
        
    Returns:
    true_snr --- (Float) True SNR of data. 
    optimization_variables --- (List) 
    Coords_flat --- (List) Sky coordinates.
    """

    # generate astronomical signal information
    numpts, spread, a, A, c, dt, p = produce_freqs_signal_params
    freqs, ast_signal = produce_freqs_signal(numpts, spread, a, A, c, dt, p)

    # true data information (projected into network)
    az_true, po_true, psi_true, geocent, coord, true_keys = true_params
    true_modes = dict.fromkeys(true_keys, ast_signal)
    data = detectors.project(freqs, geocent, az_true, po_true, psi_true, coord=coord, **true_modes)
    
    if isinstance(detectors, Network):
        true_snr = detectors.snr(freqs, geocent, az_true, po_true, psi_true, coord=coord, **true_modes)
    elif isinstance(detectors, Detector):
        true_snr = detectors.snr(freqs, data)
    else:
        raise Exception('network needs to be either a Detector or Network instance')

    # pack optimization variables
    if variables == None:
        optimization_variables = [a, A, c, detectors, freqs, geocent, data, coord, strain_keys]
    else: 
        optimization_variables = [a, A, c, detectors, freqs, geocent, data, coord, strain_keys, *variables]

    # generate strain signal sky coordinate pairs 
    azimuths, poles = az_po_meshgrid(num, coord)
    Azimuths, Poles = np.meshgrid(azimuths, poles, indexing='ij')
    Coords_flat = list(zip(Azimuths.flatten(), Poles.flatten()))

    return true_snr, optimization_variables, Coords_flat


def find_max_params(max_filter, list_results, Coords_flat, brute_params, optimization_variables, finish=True):
    """
    Helper for true_coords_cf. 
    Calculates parameters corresponding to the maximum match found through optimization.
    """
    max_skyindex = np.where(list_results == max_filter)
    max_sky_coords = [Coords_flat[i] for i in max_skyindex[0]] # need [0] bc np.where returns (array,)

    if len(max_sky_coords) == 1:
        max_az, max_po = max_sky_coords[0] 
        max_vars = [brute_max(*brute_params, *optimization_variables, max_az, max_po, finish=finish)[0]]
    else:
        max_vars = [brute_max(*brute_params, *optimization_variables, max_azim, max_pole, finish=finish)[0] 
                    for max_azim, max_pole in max_sky_coords]

    return max_sky_coords, max_vars

#----------------------------------------------------------------------------------------------
### multiprocessing over true sky coordinates azimuth, pole, and detector locations
# outer layer
# Note to self: parallelize at the outer layer for true sky coordinates azimuth, pole too if 
# bumping up resolution of true azimuth, true pole
# UNTESTED

def outer_helper(*args):
    """
    For main_cf. Returns function one_variable_filter_mp with just one argument (coordinates=azimuth, pole), for use in map.
    
    Parameters: 
    *args --- (List) [function, ranges, numpoints, finish_func, a, A, c, t0_true, phi0_true, true_psi, true_keys,
            freqs, geocent, coord, keys, t0]
    """
    helper_function = partial(brute_max, *args)
    
    return helper_function