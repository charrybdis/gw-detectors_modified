import numpy as np
from scipy.optimize import shgo, brute
from gwdetectors.detector.utils import inner_product
import warnings

#------------------------------------------------------------------------------------------------
# general functions (not optimization)

def ft_sine_Gaussian(w, a, A, c, dt=0, p=0): 
    """
    Fourier Transform of sinusoidal Gaussian pulse
    
    Parameters: 
    w --- (Number or 1D array) independent variable frequency (Hz) 
    a --- (Number) frequency of sinuisoidal component
    A --- (Number) amplitude of Gaussian envelope 
    c --- (Number) standard deviation of Gaussian envelope
    
    dt --- (Number) time shift, default 0 
    p --- (Number) phase shift, default 0
    """
    
    return A*c*np.sqrt(2*np.pi) * np.cos(p) * np.exp(-2*np.pi*1j*dt*w - (np.pi**2 * c**2 * (w-a)**2))


def produce_freqs_signal(numpts, spread, a, A, c, dt=0, p=0):
    """
    Creates a sine Gaussian signal in the frequency domain. 
    
    Parameters:
    numpts --- (Int) number of points in frequency array
    spread --- (Int) factor to multiple the standard deviation of the Gaussian peak in the frequency domain by, 
        determines the range of the frequency array
    a, A, c, dt, p --- (Number x5) parameters for ft_sine_Gaussian
    
    Returns:
    freqs --- (1D Array, length numpts) relevant frequencies centering on the middle of the peak
    ast_signal --- (1D Array, length numpts) sine Gaussian signal in frequency domain, calculated at freqs
    """

    ft_std = 1/(2 * np.pi * c) # New standard deviation of Gaussian peak in frequency domain 
    width = spread*ft_std # Width of frequency peak
    
    freqs= np.linspace(a-width, a+width, numpts)
    ast_signal = ft_sine_Gaussian(freqs, a, A, c, dt, p)
    
    return freqs, ast_signal


def az_po_meshgrid(res, coord):
    """
    Returns appropriate arrays of sky coordinate based on the coordinate system. 
    
    Parameters: 
    res --- (Int) resolution of the grid
    coord --- (String) either 'geographic' or 'celestial'
    
    Returns: 
    azimuths, poles --- (Tuple of 1D arrays of length true_res) arrays of azimuth, pole coordinates
    """
    
    if coord =='geographic':
        azimuths = np.linspace(-np.pi, np.pi, res)
        poles = np.flip(np.linspace(0, np.pi, res))
    if coord =='celestial':
        azimuths = np.linspace(0, 24, res)
        poles = np.linspace(-90, 90, res)
    
    return azimuths, poles


def calculate_snr(detector_s, num, coord, freqs, psi_true, geocent, kwargs={}):
    """
    Calculates optimal SNR over a grid for polarizations and signals given in kwargs for single detectors and networks.
    
    Parameters: 
    detector_s --- (Network or Detector) 
    num --- (Int) resolution of grid
    coord --- (String) 'geographic' or 'celestial' 
    freqs --- (1D Array) frequencies
    psi_true --- (Number) True polarization angle
    geocent --- (Number) geocent time
    kwargs --- (Dict) Keys are polarization modes, 'hp', 'hx',... 
        Values are 1D arrays of length freqs representing the signal in the frequency domain. 
    """ 

    azimuths, poles = az_po_meshgrid(num, coord)

    snr = np.zeros((num-1,num-1))

    for i, az in enumerate(azimuths[1:]):
        for j, po in enumerate(poles[1:]):
            if isinstance(detector_s, Detector):
                strain = detector_s.project(freqs, geocent, az, po, psi_true, coord='geographic', **kwargs)
                snr[i,j] = detector_s.snr(freqs, strain)
            if isinstance(detector_s, Network): 
                snr[i,j] = detector_s.snr(freqs, geocent, az, po, psi_true, 
                                          coord='geographic', 
                                          **kwargs)
    return snr


def geodetic_to_ECEF(lat, lon, h): 
    """
    Converts (Latitude, Longitude, Altitude) geodetic coordinates to Earth-fixed frame coordinates (X, Y, Z). 
    
    Parameters: 
    lat --- (Number) latitude
    lon --- (Number) longitude
    h --- (Number) height/altitude
    
    Returns: 
    X, Y, Z --- (Number, Number, Number)
    """
    a = 6378137.0 # equatorial radius of earth
    b = 6356752.3 # polar radius of earth
    
    cosPhi = np.cos(lat)
    sinPhi = np.sin(lat)
    cosLda = np.cos(lon)
    sinLda = np.sin(lon)
    
    N = a ** 2 / np.sqrt((a**2 * cosPhi**2) + (b**2 * sinPhi**2))
    
    X = (N + h)* cosPhi * cosLda
    Y = (N + h) * cosPhi * sinLda
    Z = ((b**2/a**2) * N + h) * sinPhi
    
    return X, Y, Z

#-------------------------------------------------------------------------------------------------------------------------
# optimization input functions

def filter_3(variables, a, A, c, detector_s, freqs, geocent, data, coord, keys, azim, pole): 
    """
    Calculates the real component of the network filter given injected data and a projected strain. 
    Takes all 3 variables psi, t0, and phi0. 
    
    Parameters:
    a, A, c --- (Number x3) Parameters for sine Gaussian pulse
    detector_s --- (Network) network of detectors. #try to make this work for single detectors too later? right now modfilter 
        only exists for networks. 
    freqs --- (1D array) Frequencies, should be result of produce_freqs_signal
    geocent --- (Number) geocent time
    
    data --- ([X1, X2,...] where 1, 2,... represent detectors in the network, and X1, X2,... are complex 1D arrays 
        of length frequency corresponding to the response of each detector). Result of network.project with injected variables
    coord --- (String) 'geographic' or 'celestial'
    keys --- (List of strings, ['hp', 'hx',...]) polarization modes to recover true signal with. 
    azim --- (Number) azimuthal coordinate of strain origin
    pole --- (Number) polar coordinate of strain origin
    
    Returns:
    fil --- (Int) Quadrature sum of the filter response in each detector. 
    """
    psi, t0, phi0 = variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0)
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector_s.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = detector_s.modfilter(freqs, data, proj_strain)
    
    return fil


def filter_2a(variables, a, A, c, detector_s, freqs, geocent, data, coord, keys, t0, azim, pole): 
    """
    Function to optimize over 2 variables: psi, phi0. Saves time by assuming t0 is the same as for the injected signal.
    Calculates the real component of the network filter given injected data and a projected strain.
    
    Parameters:
    See filter_3. 
    t0 --- (Number) t0 from ft_sine_Gaussian
    """
    psi, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector_s.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)

    fil = detector_s.modfilter(freqs, data, proj_strain) 
    
    return fil


def filter_2a_det(variables, a, A, c, detector_s, freqs, geocent, data, coord, keys, t0, azim, pole): 
    """
    filter_2a, but for a single detector. 
    """
    psi, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector_s.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)

    fil = detector_s.filter(freqs, data, proj_strain).real 
    
    return fil


def filter_2b(variables, a, A, c, network, freqs, geocent, data, coord, keys, p, azim, pole): 
    """
    Function to optimize over 2 variables: psi, t0. 
    Calculates the quadrature NORM of each filter response in the network given injected data and a projected strain. Implicitly
    maximizes over phi0 analytically. 
    
    Parameters: 
    p --- (Number) need to give a value for phi0
    """
    psi, t0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain)
    
    return fil


def filter_1(psi, a, A, c, network, freqs, geocent, data, coord, keys, p, t0, azim, pole): 
    """
    Calculates the filter over one primary independent variable psi only. Implicitly maximizes over phi0 analytically, 
    can manually input the expected, "true" value of t0. Fastest of the optimization input functions.
    
    Parameters:
    p, t0 --- (Number x2) phi0, t0 from ft_sine_Gaussian
    """

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain)
    
    return fil

#-------------------------------------------------------------------------------------------------------------------------
# optimization functions

def brute_max(function, ranges, numpoints, finish_func, *args):
    """
    Maximizing over general function for ranges of variables with the scipy brute function. 
    
    Parameters:
    function --- function to maximize over
    ranges, numpoints, finish_func --- parameters for scipy brute function
    *args --- arguments for 'function' that are not the variables being optimized over
    
    Returns:
    optimization_result[0], -optimization_result[1] --- variables at maximum, maximum
    """
    fun=lambda variables: -function(variables, *args)
    
    optimization_result = brute(fun, ranges=ranges, Ns=numpoints, finish=finish_func, full_output=True)
    
    return optimization_result[0], -optimization_result[1]


def shgo_max(function, bounds, *args):
    """
    Maximizing over general function for ranges of variables with the scipy shgo function. 
    """
    fun=lambda variables: -function(variables, *args)
    
    optimization_result = shgo(fun, bounds)
    
    if optimization_result.success == False: 
        warnings.warn("Optimization failed")

    return -optimization_result.fun


def ift_max(network, nums, a, A, c, freqs, geocent, coord, keys, azim, pole):
    """
    DOESN'T WORK YET
    """
    
    psi_num, phi_num = nums # unpack number of grid points wanted for psi and phi respectively
    
    # initialize variables
    max = 0
    max_location = (0, 0, 0)
    
    # create appropriate ranges for psi and phi
    psi_range = np.linspace(0, np.pi, psi_num)
    phi_range = np.linspace(0, 2*np.pi, phi_num)
    
    psd = network.psd
   
    for psi in psi_range:
        for phi in phi_range:
            strain_signal = ft_sine_Gaussian(freqs, a, A, c, 0, phi) # t0 set to 0 here
            modes = dict.fromkeys(keys, strain_signal)
            strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
            filter_t = network.ftfilter(freqs, data, strain)
            rho_t = np.max(filter_t)
            t0 = np.where(filter_t == np.max(filter_t))
            
            if rho_t > max:
                max = rho_t
                max_location = (psi, phi, t0)

    return max_location, max

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
            max_vars = Functions.brute_max(*brute_params, *optimization_variables,
                                           max_az, max_po)[0]
        else:
            max_vars = [Functions.brute_max(*brute_params, *optimization_variables,
                                                 max_az, max_po) for max_az, max_po in max_sky_coords]

        run_results = {'pole': true_po, 'azim':true_az, 'psi': true_psi, 
                           'filter':max_filter, 'match':rho_match, 'max_sky_coords':max_sky_coords, 'max_vars':max_vars}
        current_dict["_".join(strain_keys) + "-" + f"{i}"] = run_results
    return current_dict
