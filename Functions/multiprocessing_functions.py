import numpy as np
import warnings
from .general_functions import * 
from .optimization_functions import *

#----------------------------------------------------------------------------------------------
# multiprocessing
# can't apply this to optimization over t0, psi because the optimization function is an imported module and the multiprocessing
# must be run in __main__

from functools import partial
import concurrent.futures


def one_variable_filter_mp(function, ranges, numpoints, finish_func, optimization_variables, coordinates): 
    """
    Essentially the same as brute_max, but with azimuth, pole as a single coordinate tuple.
    
    Parameters:
    function, ranges, numpoints, finish_func --- parameters in brute
    optimization_variables --- (List) variables for the optimization function
    """
    az, po = coordinates
    full_func = brute_max(function, ranges, numpoints, finish_func, *optimization_variables, az, po)[1]
    
    return full_func


def one_variable_mp(*args):
    """
    For main_cf. Returns function one_variable_filter_mp with just one argument (coordinates=azimuth, pole), for use in map.
    
    Parameters: 
    *args --- all the parameters for one_variable_filter_mp, EXCEPT coordinates
    """
    one_variable = partial(one_variable_filter_mp, *args)
    
    return one_variable


def main_cf(workers, num, Coords_flat, *args):
    """
    Runs one_variable_mp over Coords_flat with concurrent.futures. 
    Needs to be called in main with if __name__ =='__main__'!!
    
    Parameters:
    workers --- (Int or None) max number of workers, set to None for automatic assignment
    num --- (Int) number of grid points in original azimuth, pole arrays
    Coords_flat --- (List of tuples (azimuth, pole)) List of coordinates in (azimuth, pole) format. 
    *args --- *args for one_variable_mp
    """
    
    one_variable = one_variable_mp(*args)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(one_variable, Coords_flat)
    list_results = np.array(list(results))
    filter_grid = np.reshape(list_results, (num-1, num-1))
    
    return filter_grid

#----------------------------------------------------------------------------------------------
# more complicated functions with multiprocessing

def produce_optimization_params(network, produce_freqs_signal_params, true_params, strain_keys, num): 
    """
    Creates all the information needed to optimize the filter response over psi and phi and calculate it over a grid.
    NOTE: if changing the optimization input function (e.g. filter_3 vs filter_2a vs...), need to append t0, phi0, or 
    whatever may apply to optimization_variables
    
    Parameters: 
    network --- (Network) instance of Network containing Detectors
    produce_freqs_signal_params --- (List) parameters for produce_freqs_signal function in Functions
    true_params --- (List) parameters for "true" data; azimuth, pole, psi, geocent_time, coord, {hp:signal, hx:signal,....} 
    brute_params --- (List) parameters for scipy brute function
    strain_keys --- (List of strings) ['hp', 'hx',...]
    num --- (Int) number of grid points for sky 
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
    optimization_variables = [a, A, c, network, freqs, geocent, data, coord, strain_keys, 0] # need 0 if using filter_2a

    # generate sky coordinate pairs
    azimuths, poles = az_po_meshgrid(num, coord)
    Azimuths, Poles = np.meshgrid(azimuths[1:], poles[1:], indexing='ij')
    Coords_flat = list(zip(Azimuths.flatten(), Poles.flatten()))

    return true_snr, optimization_variables, Coords_flat

#----------------------------------------------------------------------------------------------
# iteration with multiprocessing

def detector_iterate_cf(new_detectors, network_initial, 
                        produce_freqs_signal_params, true_params, strain_keys, num, brute_params,
                        full_output=False):
    """
    Same as detector_iterate but with concurrent futures. For testing on server without Torque. 
    Iterates over a set of new detectors to find the network which provides the smallest match. 
    
    Parameters:
    new_detectors --- (List) 
    network_initial --- (Network) base network
    
    Returns: 
    match --- (Number) match value
    best_detector --- (Detector) best detector to append to network_initial in order to minimize the match
    """
    
    match = 1
    best_detector = new_detectors[0]
 
    for new_detector in new_detectors: 
        new_network = Network(*network_initial.detectors, new_detector)
        
        print(type(new_network))

        true_snr, optimization_variables, Coords_flat=produce_optimization_params(new_network, produce_freqs_signal_params,
                                                                                  true_params, strain_keys, num)
        list_results = main_cf(None, num, Coords_flat, *brute_params, optimization_variables)

        filter_grid = np.reshape(list_results, (num-1, num-1))
        
        max_filter = np.max(filter_grid)
        rho_match = max_filter / true_snr
        
        if rho_match < match: 
            match = rho_match
            best_detector = new_detector
            
    if full_output == True: 
        pass
            
    return match, best_detector


def true_coords_iterate_cf(true_coords, true_psi, geocent, coord, true_keys, 
                        network, produce_freqs_signal_params, strain_keys, num, 
                        brute_params,
                        full_output=False):
    
    current_dict={}

    for i, signal_coord in enumerate(true_coords): 
        true_az, true_po, true_psi = signal_coord
        true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]
        
        true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                                  true_params, strain_keys, num)
        list_results = main_cf(None, num, Coords_flat, *brute_params, optimization_variables)
        
        filter_grid = np.reshape(list_results, (num-1, num-1))
        
        # get parameters, save results to dictionary
        max_filter = np.max(filter_grid)
        rho_match = max_filter / true_snr
        
        max_skyindex = np.where(list_results == np.max(list_results))
        max_sky_coords = [Coords_flat[i] for i in max_skyindex[0]] # need the [0] bc np.where automatically returns (array,)
        max_az, max_po = max_sky_coords[0]
        
        if len(max_sky_coords) == 1:
            max_vars = brute_max(*brute_params, *optimization_variables,
                                 max_az, max_po)[0]
        else:
            max_vars = [brute_max(*brute_params, *optimization_variables,
                                  max_az, max_po) for max_az, max_po in max_sky_coords]

        run_results = {'pole': true_po, 'azim':true_az, 'psi': true_psi, 
                           'filter':max_filter, 'match':rho_match, 'max_sky_coords':max_sky_coords, 'max_vars':max_vars}
        current_dict["_".join(strain_keys) + "-" + f"{i}"] = run_results
    return current_dict

#----------------------------------------------------------------------------------------------

def true_coords_cf(signal_coord, true_psi, geocent, coord, true_keys,
                   network, produce_freqs_signal_params, strain_keys, 
                   workers, num, brute_params, 
                   full_output=True):
    """
    Calculates optimization over true coordinates for the data. Uses mpi4py.futures.
    Allows for moving the iteration loop outside the function so the results of each iteration can be saved separately.
    
    Parameters:
    signal_coord --- (Tuple) (azimuth, pole) coordinates
    full_output --- (Bool) Set to true to return max_sky_coords and max_vars information. Otherwise, other than the true 
        coordinates of the injected data, only the match and the maximum filter will be included. 
    
    Returns: 
    run_results --- (Dict) Match results and parameters for signal_coord
    """
    
    true_az, true_po, true_psi = signal_coord # unpack coordinates
    true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]
    true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                                  true_params, strain_keys, num)
    list_results = main_cf(workers, num, Coords_flat, *brute_params, optimization_variables) # optimization
    filter_grid = np.reshape(list_results, (num-1, num-1))
    
    max_filter = np.max(filter_grid)
    rho_match = max_filter / true_snr
    
    if full_output == True:
        max_skyindex = np.where(list_results == np.max(list_results))
        max_sky_coords = [Coords_flat[i] for i in max_skyindex[0]] # need the [0] bc np.where automatically returns (array,)
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