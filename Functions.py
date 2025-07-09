import numpy as np
from scipy.optimize import Bounds, shgo, brute, dual_annealing, fmin
from gwdetectors.detector.utils import inner_product
import warnings

#------------------------------------------------------------------------------------------------

def ft_sine_Gaussian(w, a, A, c, dt=0, p=0): 
    """
    Fourier Transform of sinusoidal Gaussian pulse
    
    Parameters: 
    w --- independent variable frequency (Hz) 
    a --- frequency of sinuisoidal component
    A --- amplitude of Gaussian envelope 
    c --- standard deviation of Gaussian envelope
    
    dt --- time shift 
    p --- phase shift
    """
    
    return A*c*np.sqrt(2*np.pi) * np.cos(p) * np.exp(-2*np.pi*1j*dt*w - (np.pi**2 * c**2 * (w-a)**2))


def produce_freqs_signal(numpts, spread, a, A, c, dt=0, p=0):
    """
    Creates a sine Gaussian signal in the frequency domain. 
    
    Parameters:
    numpts --- number of points in frequency array
    spread --- factor to multiple the standard deviation of the Gaussian peak in the frequency domain by, determines the range
    of the frequency array
    a, A, c, dt, p --- parameters for ft_sine_Gaussian
    
    Returns:
    freqs --- array of frequencies
    ast_signal --- sine Gaussian signal in frequency domain, calculated at freqs
    """

    ft_std = 1/(2 * np.pi * c) # New standard deviation of Gaussian peak in frequency domain 
    width = spread*ft_std # Width of frequency peak
    
    freqs= np.linspace(a-width, a+width, numpts)
    ast_signal = ft_sine_Gaussian(freqs, a, A, c, dt, p)
    
    return freqs, ast_signal

#-------------------------------------------------------------------------------------------------------------------------
# optimization

def filter_3(variables, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys): 
    """
    Calculates the real component of the network filter given injected data and a projected strain. 
    Takes all 3 variables psi, t0, and phi0. 
    
    Parameters:
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    """
    psi, t0, phi0 = variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0)
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.modfilter(freqs, data, proj_strain).real
    
    return fil


def filter_2a(variables, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys, t0): 
    """
    Function to optimize over 2 variables: psi, phi0. 
    Calculates the real component of the network filter given injected data and a projected strain. 
    
    Parameters:
    a, A, c --- Parameters for sine Gaussian pulse
    
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    
    keys --- ['hp', 'hx',...]
    t0 --- t0 from ft_sine_Gaussian
    """
    psi, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.modfilter(freqs, data, proj_strain).real
    
    return fil


def filter_2b(variables, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys, p): 
    """
    Function to optimize over 2 variables: psi, t0. 
    Calculates the quadrature NORM of each filter response in the network given injected data and a projected strain. Implicitly
    maximizes over phi0 analytically. 
    
    Parameters:
    a, A, c --- Parameters for sine Gaussian pulse
    
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    
    keys --- ['hp', 'hx',...]
    p --- phi0
    """
    psi, t0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain)
    
    return fil


def filter_1(psi, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys, p, t0): 
    """
    Calculates the filter over one primary independent variable psi only. Implicitly maximizes over phi0 analytically.
    
    Parameters:
    a, A, c --- Parameters for sine Gaussian pulse
    
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    
    keys --- ['hp', 'hx',...]
    p, t0 --- phi0, t0 from ft_sine_Gaussian
    """

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain)
    
    return fil


def brute_max(function, ranges, numpoints, finish_func, *args):
    """
    Maximizing over general function for ranges of variables with the scipy brute function. 
    
    Parameters:
    function --- function to maximize over
    ranges, numpoints, finish_func --- parameters for brute
    *args --- arguments for 'function' that are not the variables being optimized over
    
    Returns:
    variables at maximum, maximum
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


def ift_max(network, nums, a, A, c, freqs, geocent, azim, pole, coord, keys):
    
    psi_num, phi_num = nums # unpack number of grid points wanted for psi and phi respectivel
    
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


def calculate_snr(detector_s, num, freqs, psi_true, geocent, kwargs={}):
    """Calculates optimal SNR over a grid for polarizations and signals given in kwargs for single detectors and networks.""" 

    azimuths = np.linspace(-np.pi, np.pi, num)
    poles = np.flip(np.linspace(0, np.pi, num))

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

#----------------------------------------------------------------------------------------------
# multiprocessing
# can't apply this to optimization over t0, psi because the optimization function is an imported module and the multiprocessing
# must be run in __main__

from functools import partial
import concurrent.futures


def one_variable_filter_mp(filter_func, ranges, npts, finish_fun, 
                        a, A, c, network, freqs, geocent, data, coord, keys, phi, coordinates): 
    """
    Essentially the same as brute_max, but with azimuth, pole as a single coordinate tuple.
    
    Parameters:
    filter_func, ranges, npts, finish_fun --- explicit parameters for brute_max
    *args --- parameters that go into filter_func
    """
    az, po = coordinates
    full_func = brute_max(filter_func, ranges, npts, finish_fun, a, A, c, 
                               network, freqs, geocent, data, az, po, coord, keys, 0)[1]
    
    return full_func


def one_variable_mp(*args):
    """
    Returns function one_variable_filter_mp with just one argument (coordinates), for use in map.
    
    Parameters: 
    *args --- all the parameters for one_variable_filter_mp, EXCEPT coordinates
    """
    one_variable = partial(one_variable_filter_mp, *args)
    
    return one_variable


def main(workers, num, Coords_flat, *args):
    """
    Runs one_variable_mp over Coords_flat with multiprocessing. Needs to be called in main with if __name__ ==
    '__main__'!!
    
    Parameters:
    workers --- max number of workers, set to None for automatic 
    num --- number of grid points in original azimuth, pole arrays
    Coords_flat --- list of coordinates in (azimuth, pole) format. Needs to be 1D.
    *args --- arguments for one_variable_mp
    """

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(one_variable_mp(*args), Coords_flat)
    list_results = np.array(list(results))
    filter_grid = np.reshape(list_results, (num-1, num-1))
    
    return filter_grid
