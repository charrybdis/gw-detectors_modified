import numpy as np
from scipy.optimize import brute, minimize, Bounds
from gwdetectors.detector.utils import inner_product
import warnings
from .general_functions import *

#-------------------------------------------------------------------------------------------------------------------------
# optimization input functions, numbered based on the number of variables they optimize over
# note that the coordinates being calculated over using concurrent.futures must be the final two arguments of the function

def filter_5(variables, a, A, c, detector_s, freqs, geocent, coord, keys, 
             true_t0, true_psi, true_phi0, true_keys, true_azim, true_pole): 
    """
    Calculates the network filter given injected data and a projected strain. 
    Takes variables psi, t0, and phi0, and strain sky coordinates azimuth, pole. 
    """
    psi, t0, phi0, azim, pole = variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0)
    true_signal = ft_sine_Gaussian(freqs, a, A, c, true_t0, true_phi0)
    
    modes = dict.fromkeys(keys, strain_signal) 
    true_modes = dict.fromkeys(true_keys, true_signal)
    proj_strain = detector_s.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    data = detector_s.project(freqs, geocent, true_azim, true_pole, psi, coord=coord, **true_modes)
    
    fil = detector_s.modfilter(freqs, data, proj_strain)
    
    return fil


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
    azim --- (Number) azimuthal coordinate of strain origin
    pole --- (Number) polar coordinate of strain origin
    data --- ([X1, X2,...] where 1, 2,... represent detectors in the network, and X1, X2,... are complex 1D arrays 
        of length frequency corresponding to the response of each detector). Result of network.project with injected variables
    coord --- (String) 'geographic' or 'celestial'
    keys --- (List of strings, ['hp', 'hx',...]) polarization modes to recover true signal with. 
    
    Returns:
    fil --- (Int) Quadrature sum of the filter response in each detector. 
    """
    psi, t0, phi0 = variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0)
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector_s.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = detector_s.modfilter(freqs, data, proj_strain)
    
    return fil


def filter_3_det(variables, a, A, c, detector, freqs, geocent, data, coord, keys, azim, pole): 
    """
    filter_3, but for a single detector. 
    """
    psi, t0, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)

    fil = detector.filter(freqs, data, proj_strain).real 
    
    return fil


def filter_2a(variables, a, A, c, detector_s, freqs, geocent, data, coord, keys, t0, azim, pole): 
    """
    Function to optimize over 2 variables: psi, phi0. 
    Calculates the real component of the network filter given injected data and a projected strain.
    
    Parameters:
    t0 --- (Number) t0 from ft_sine_Gaussian
    """
    psi, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector_s.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)

    fil = detector_s.modfilter(freqs, data, proj_strain) 
    
    return fil


def filter_2a_det(variables, a, A, c, detector, freqs, geocent, data, coord, keys, t0, azim, pole): 
    """
    filter_2a, but for a single detector. 
    """
    psi, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = detector.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)

    fil = detector.filter(freqs, data, proj_strain).real 
    
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
    
    fil = network.normfilter(freqs, data, proj_strain)
    
    return fil


def filter_1(psi, a, A, c, network, freqs, geocent, data, coord, keys, p, t0, azim, pole): 
    """
    Calculates the filter over one primary independent variable psi only. Implicitly maximizes over phi0 analytically, 
    can manually input the expected, "true" value of t0.
    """

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.normfilter(freqs, data, proj_strain)
    
    return fil

#-------------------------------------------------------------------------------------------------------------------------
# optimization functions

def brute_max(function, ranges, numpoints, *args, finish=False):
    """
    Maximizing over general function for ranges of variables with the scipy brute function. 
    
    Parameters:
    function --- (Function) function to maximize over
    ranges, numpoints --- (Slice), (Int) parameters for scipy brute function
    *args --- arguments for 'function' that are not the variables being optimized over
    finish --- (Bool) If False, returns result of scipy brute. If true, "polishes" result with Nelder-Mead. 
    
    Returns:
    optimization_result[0], -optimization_result[1] --- (Tuple or Number, Number) variables at maximum, maximum
    """
    fun=lambda variables: -function(variables, *args)
    
    optimization_result = brute(fun, ranges=ranges, Ns=numpoints, finish=None, full_output=True)
    
    if finish == True:
        [*r] = ranges
        spacings = []
        for i in range(len(r)): 
            start, stop, spacing = r[i]
            spacings.append(spacing)
        lbs = np.array(optimization_result[0]) - np.array(spacings)
        ubs = np.array(optimization_result[0]) + np.array(spacings)
        finished = minimize(fun, optimization_result[0], method='Nelder-Mead', tol=1e-6, bounds=Bounds(lbs, ubs))
        variables = finished.x
        result = finished.fun
        return variables, -result
    else: 
        return optimization_result[0], -optimization_result[1]


def ift_max(network, num, a, A, c, freqs, geocent, data, coord, strain_keys, strain_azim, strain_pole):
    """
    DOESN'T WORK
    """
    # initialize variables
    max_filter = 0
    
    # create appropriate ranges for psi and phi
    psi_range = np.linspace(0, 2*np.pi, num, endpoint=False)
    phi_range = np.linspace(0, 2*np.pi, num, endpoint=False)
    
    for psi in psi_range:
        for phi in phi_range:
            # for each given psi and phi combination, find the maximum t0
            strain_signal = ft_sine_Gaussian(freqs, a, A, c, 0, phi) # t0 set to 0 here
            modes = dict.fromkeys(strain_keys, strain_signal)
            strain = network.project(freqs, geocent, strain_azim, strain_pole, psi, coord=coord, **modes)
            filter_t0 = network.ftfilter(freqs, data, strain)
            rho_t0 = np.max(filter_t0)
            t0 = np.where(filter_t0 == rho_t0)[0]
            
            if rho_t0 > max_filter:
                max_filter = rho_t0
                max_location = (psi, phi, t0)

    return max_location, max_filter