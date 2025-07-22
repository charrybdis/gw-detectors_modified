from gwdetectors import *
from Functions import *
import numpy as np

from mpi4py import MPI # this needs to be imported only where torque is installed or it gives annoying warnings 
from mpi4py.futures import MPIPoolExecutor

#--------------------------------------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------------------------------------------

def detector_iterate(new_detectors, network_initial,
                     produce_freqs_signal_params, true_params, strain_keys, num, brute_params,
                     full_output=False):
    """
    Iterates over a set of new detectors to find the network which provides the smallest match. Uses mpi4py.futures.
    
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
        network = Network(*network_initial.detectors, new_detector)

        true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                                  true_params, strain_keys, num)
        list_results = main_mpi(None, num, Coords_flat, *brute_params, optimization_variables)

        filter_grid = np.reshape(list_results, (num-1, num-1))
        
        max_filter = np.max(filter_grid)
        rho_match = max_filter / true_snr
        
        if rho_match < match: 
            match = rho_match
            best_detector = new_detector
            
    if full_output == True: 
        pass
            
    return match, best_detector


def true_coords_iterate(true_coords, true_psi, geocent, coord, true_keys, 
                        network, produce_freqs_signal_params, strain_keys, num, 
                        brute_params,
                        full_output=False):
    """
    Iterates optimization over a set of true coordinates for the data. Uses mpi4py.futures.
    
    Parameters:
    true_coords --- (List) list of (azimuth, pole) coordinates
    
    Returns: 
    current_dict --- (Dict) Match results and parameters for all coordinates in true_coords
    """

    current_dict={}

    for i, signal_coord in enumerate(true_coords):
        true_az, true_po, true_psi = signal_coord
        true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]

        true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                                  true_params, strain_keys, num)
        list_results = main_mpi(None, num, Coords_flat, *brute_params, optimization_variables)
        filter_grid = np.reshape(list_results, (num-1, num-1))

        # get parameters, save results to dictionary
        max_filter = np.max(filter_grid)
        rho_match = max_filter / true_snr

        max_skyindex = np.where(list_results == np.max(list_results))
        max_sky_coords = [Coords_flat[i] for i in max_skyindex[0]] # need the [0] bc np.where automatically returns (array,)
        max_az, max_po = max_sky_coords[0]

        if len(max_sky_coords) == 1:
            max_vars = Functions.brute_max(*brute_params, *optimization_variables,
                                           max_az, max_po)[0]
        else:
            max_vars = [Functions.brute_max(*brute_params, *optimization_variables,
                                            max_az, max_po) for max_az, max_po in max_sky_coords]

        run_results = {'pole': true_po, 'azim':true_az, 'psi': true_psi,
                       'filter':max_filter, 'match':rho_match, 'max_sky_coords':max_sky_coords, 'max_vars':max_vars}
        current_dict["_".join(strain_keys) + "-" + f"{i}"] = run_results
    return current_dict

#----------------------------------------------------------------------------------------------------------------

def true_coords_(signal_coord, true_psi, geocent, coord, true_keys,
                 network, produce_freqs_signal_params, strain_keys, num, 
                 brute_params, full_output=False):
    """
    Calculates optimization over true coordinates for the data. Uses mpi4py.futures.
    Allows for moving the iteration loop outside the function so the results of each iteration can be saved separately.
    
    Parameters:
    signal_coord --- (Tuple) (azimuth, pole) coordinates
    
    Returns: 
    run_results --- (Dict) Match results and parameters for signal_coord
    """

    true_az, true_po, true_psi = signal_coord
    true_params = [true_az, true_po, true_psi, geocent, coord, true_keys]

    true_snr, optimization_variables, Coords_flat=produce_optimization_params(network, produce_freqs_signal_params,
                                                                              true_params, strain_keys, num)
    list_results = main_mpi(None, num, Coords_flat, *brute_params, optimization_variables)
    filter_grid = np.reshape(list_results, (num-1, num-1))

    # get parameters, save results to dictionary
    max_filter = np.max(filter_grid)
    rho_match = max_filter / true_snr

    max_skyindex = np.where(list_results == np.max(list_results))
    max_sky_coords = [Coords_flat[i] for i in max_skyindex[0]] # need the [0] bc np.where automatically returns (array,)
    max_az, max_po = max_sky_coords[0]

    if len(max_sky_coords) == 1:
        max_vars = Functions.brute_max(*brute_params, *optimization_variables,
                                       max_az, max_po)[0]
    else:
        max_vars = [Functions.brute_max(*brute_params, *optimization_variables,
                                        max_az, max_po) for max_az, max_po in max_sky_coords]

    run_results = {'pole': true_po, 'azim':true_az, 'psi': true_psi,
                   'filter':max_filter, 'match':rho_match, 'max_sky_coords':max_sky_coords, 'max_vars':max_vars}
    return run_results
   