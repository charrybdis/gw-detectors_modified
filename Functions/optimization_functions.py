import numpy as np
from scipy.optimize import shgo, brute
from gwdetectors.detector.utils import inner_product
import warnings
from .general_functions import *

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