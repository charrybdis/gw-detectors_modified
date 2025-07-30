import numpy as np
import warnings
from .general_functions import * 
from .optimization_functions import *
from functools import partial
import concurrent.futures

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


def main_cf(Coords_flat, *args, workers=None, finish=False):
    """
    Runs one_variable_mp over Coords_flat with concurrent.futures. 
    Needs to be called in main with if __name__ =='__main__'!!
    
    Parameters:
    workers --- (Int or None) Max number of worker processes, set to None for automatic assignment based on available processors.
    Coords_flat --- (List of tuples) List of coordinates in (azimuth, pole) format. 
    *args --- *args for one_variable_mp
    
    Returns:
    list_results --- (1D Array) Result of one_variable_mp function at each value of Coords_flat.
    """
    
    one_variable = one_variable_mp(*args, finish=finish)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(one_variable, Coords_flat)
    list_results = np.array(list(results))
    
    return list_results

#----------------------------------------------------------------------------------------------
# more complicated functions with multiprocessing

def produce_optimization_params(network, produce_freqs_signal_params, true_params, strain_keys, num, variables=None): 
    """
    Helper for true_coords_cf. 
    Creates all the information needed to optimize the filter response over psi, phi, t0, and strain sky coordinates.

    Parameters: 
    network --- (Network) instance of Network containing Detectors
    produce_freqs_signal_params --- (List) parameters for produce_freqs_signal function in Functions
    true_params --- (List) parameters for "true" data; azimuth, pole, psi, geocent_time, coord, {hp:signal, hx:signal,....} 
    brute_params --- (List) parameters for scipy brute function
    strain_keys --- (List of strings) ['hp', 'hx',...]
    num --- (Int) number of grid points for sky
    variables --- (List) Defaults to None, corresponding to optimization over all 3 variables (psi, phi, t0). Otherwise, 
        should be a list containing the fixed values of the variables not being optimized over. 
        
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
    data = network.project(freqs, geocent, az_true, po_true, psi_true, coord=coord, **true_modes)
    true_snr = network.snr(freqs, geocent, az_true, po_true, psi_true, coord=coord, **true_modes)

    # pack optimization variables
    if variables == None:
        optimization_variables = [a, A, c, network, freqs, geocent, data, coord, strain_keys]
    else: 
        optimization_variables = [a, A, c, network, freqs, geocent, data, coord, strain_keys, *variables] # need 0 if using filter_2a

    # generate strain signal sky coordinate pairs 
    azimuths, poles = az_po_meshgrid(num, coord)
    Azimuths, Poles = np.meshgrid(azimuths, poles, indexing='ij')
    Coords_flat = list(zip(Azimuths.flatten(), Poles.flatten()))

    return true_snr, optimization_variables, Coords_flat


def true_coords_cf_grid(signal_coord, true_psi, geocent, coord, true_keys,
                        network, produce_freqs_signal_params, strain_keys, 
                        num, brute_params, 
                        variables=None, 
                        workers=None, 
                        finish=False):
    """
    Calculates optimization for a set of true coordinates for the data over a grid of strain sky coordinates. 
    Uses concurrent.futures. Returns the filter response as a grid. 
    
    Returns: 
    true_snr, filter_grid --- (Float, Array) 
    true_snr, optimization_variables, Coords_flat, list_results --- (Float, List, List, List)
    """
    
    true_az, true_po, true_psi = signal_coord # unpack coordinates
    true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]
    true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                              true_params, strain_keys, num, 
                                                                              variables=variables)
    # optimization
    list_results = main_cf(Coords_flat, *brute_params, optimization_variables, 
                           workers=workers, finish=finish)
    
    filter_grid = np.reshape(list_results, (num, num))
    return true_snr, filter_grid


def true_coords_cf(signal_coord, true_psi, geocent, coord, true_keys,
                   network, produce_freqs_signal_params, strain_keys, 
                   num, brute_params, 
                   variables=None, 
                   workers=None, 
                   finish=False,
                   full_output=True):
    """
    Calculates optimization for a set of true coordinates for the data over a grid of strain sky coordinates. 
    Uses concurrent.futures.

    Parameters:
    signal_coord --- (Tuple) (azimuth, pole) coordinates. NO PSI. 
    full_output --- (Bool) Set to true to return max_sky_coords and max_vars information. Otherwise, other than the true 
        coordinates of the injected data, only the match and the maximum filter will be included. 
    
    Returns:
    run_results --- (Dict) Maximum match and filter and parameters for true signal. 
        If full_output is true, also contains variables that give maximum match.
    """
    true_az, true_po, true_psi = signal_coord # unpack coordinates
    true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]
    true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                              true_params, strain_keys, num, 
                                                                              variables=variables)
    # optimization
    list_results = main_cf(Coords_flat, *brute_params, optimization_variables, 
                           workers=workers, finish=finish)
    
    # calculate maximum, match
    max_filter = np.max(list_results)
    rho_match = max_filter / true_snr
    
    # find parameters
    if full_output == True:
        max_skyindex = np.where(list_results == np.max(list_results))
        max_sky_coords = [Coords_flat[i] for i in max_skyindex[0]] # need [0] bc np.where returns (array,)
        max_az, max_po = max_sky_coords[0] # need this in case len(max_sky_coords) == 1

        if len(max_sky_coords) == 1:
            max_vars = brute_max(*brute_params, *optimization_variables,
                                 max_az, max_po)[0]
        else:
            max_vars = [brute_max(*brute_params, *optimization_variables,
                                  max_az, max_po)[0] for max_az, max_po in max_sky_coords]

        run_results = {'true pole': true_po, 'true azim':true_az, 'true psi': true_psi,
                       'filter':max_filter, 'match':rho_match, 'max_sky_coords':max_sky_coords, 'max_vars':max_vars}
    else:
        run_results = {'true pole': true_po, 'true azim':true_az, 'true psi': true_psi, 
                       'filter':max_filter, 'match':rho_match}

    return run_results

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


def main_outer_cf(dim, existing_network, new_detectors, true_azims, true_poles, *args, workers=None):
    """
    Parameters:
    workers --- (Int or None) max number of workers, set to None for automatic assignment
    dim --- (Int) N, length of new_detectors, true_azimuth, true_pole arrays
    new_detectors --- (List of length N) 
    true_azims --- (List of length N)
    true_poles --- (List of length N) 
    *args --- *args for helper_function
    """
    new_detectors, true_azims, true_poles = variables
    
    helper = outer_helper(*args)
    
    existing_detectors = existing_network.detectors
    new_networks = [Network(*existing_detectors, new_detector) for new_detector in new_detectors]

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(helper, new_networks, true_azims, true_poles)
    list_results = np.array(list(results))
    grid = np.reshape(list_results, (dim, dim, dim))
    
    return grid