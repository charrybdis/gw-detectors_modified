import numpy as np
from scipy.optimize import Bounds, shgo, brute, dual_annealing, fmin
import warnings

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

    ft_std = 1/(2 * np.pi * c) # New standard deviation of Gaussian peak in frequency domain 
    width = spread*ft_std # Width of frequency peak
    
    freqs= np.linspace(a-width, a+width, numpts)
    ast_signal = ft_sine_Gaussian(freqs, a, A, c, dt, p)
    
    return freqs, ast_signal


def filter_1(variables, a, A, c, network, freqs, geocent, data, azim, pole, coord): 
    """
    Function for use in general_max, calculates the real component of the network filter given injected data and a projected
    strain. 
    
    Parameters:
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    """
    psi, t0, phi0 = variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0)
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, hx=strain_signal)
    
    fil = network.mfilter(freqs, data, proj_strain).real
    
    return fil


def filter_2(variables, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys): 
    """
    Function for use in general_max in order to optimize over psi, phi0. Calculates the real component of the network filter given injected data and a projected strain. 
    
    Parameters:
    a, A, c --- Parameters for sine Gaussian pulse
    
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    
    keys --- ['hp', 'hx',...]
    """
    psi, phi0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, phi0) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain).real
    
    return fil


def filter_3(variables, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys, p): 
    """
    Function for use in general_max in order to optimize over psi, phi0. Calculates the norm of the network filter given injected data and a projected strain. 
    
    Parameters:
    a, A, c --- Parameters for sine Gaussian pulse
    
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    
    keys --- ['hp', 'hx',...]
    """
    psi, t0 = variables # unpack variables

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain)
    
    return fil


def filter_4(psi, a, A, c, network, freqs, geocent, data, azim, pole, coord, keys, p, t0): 
    """
    Optimizing over psi only. 
    
    Parameters:
    a, A, c --- Parameters for sine Gaussian pulse
    
    data --- result of network.project with injected variables
    azim --- azimuthal coordinate of strain origin
    pole --- polar coordinate of strain origin
    coord --- 'geographic' or 'celestial'
    
    keys --- ['hp', 'hx',...]
    """

    strain_signal = ft_sine_Gaussian(freqs, a, A, c, t0, p) # create template strain
    
    modes = dict.fromkeys(keys, strain_signal) 
    proj_strain = network.project(freqs, geocent, azim, pole, psi, coord=coord, **modes)
    
    fil = network.mfilter(freqs, data, proj_strain)
    
    fil_norm = np.sqrt(fil.real**2 + fil.imag**2)
    
    return fil_norm


def brute_max(function, ranges, numpoints, finish_func, *args):
    """
    Maximizing over general function for ranges of variables with the scipy brute function. 
    
    For filter_1: 
    Parameters:
    *args --- a, A, c, network, freqs, geocent, data, azim, pole, coord 
    """
    fun=lambda variables: -function(variables, *args)
    
    optimization_result = brute(fun, ranges=ranges, Ns=numpoints, finish=finish_func, full_output=True)
    
    # returns variables of minimum, minimum 
    return optimization_result[0], -optimization_result[1]


def shgo_max(function, bounds, *args):
    """
    Maximizing over general function for ranges of variables with the scipy shgo function. 
    
    For filter_1: 
    Parameters:
    *args --- a, A, c, network, freqs, geocent, data, azim, pole, coord 
    """
    fun=lambda variables: -function(variables, *args)
    
    optimization_result = shgo(fun, bounds)
    
    if optimization_result.success == False: 
        warnings.warn("Optimization failed")

    return -optimization_result.fun


def grid_filter_1_max(num, ranges, *args):
    """
    Calculates filter_1_max over a grid of possible sky origin positions for the strain. 
    """
    if coord == 'geographic': 
        azimuths = np.linspace(-np.pi, np.pi, num)
        poles = np.flip(np.linspace(0, np.pi, num))
        
    if coord == 'celestial':
        azimuths = np.linspace(0, 24, num)
        poles = np.linspace(np.pi/2, np.pi/2, num)
    
    fil = np.zeros((num-1,num-1))
    for i, az in enumerate(azimuths[1:]): 
        for j, po in enumerate(poles[1:]):  
            fil[j,i] = filter_1_max(ranges, network, freqs, geocent, proj_true, az, po)

    return fil


def calculate_snr(detector_s, num, freqs, psi_true, geocent, kwargs={}):
    """Calculates optimal SNR over a grid for polarizations and signals given in kwargs for single detectors and networks.""" 

    azimuths = np.linspace(-np.pi, np.pi, num)
    poles = np.flip(np.linspace(0, np.pi, num))

    snr = np.zeros((num-1,num-1))

    for i, az in enumerate(azimuths[1:]):
        for j, po in enumerate(poles[1:]):
            if isinstance(detector_s, Detector):
                strain = detector_s.project(freqs, geocent, az, po, psi_true, coord='geographic', **kwargs)
                snr[j,i] = detector_s.snr(freqs, strain)
            if isinstance(detector_s, Network): 
                snr[j,i] = detector_s.snr(freqs, geocent, az, po, psi_true, 
                                          coord='geographic', 
                                          **kwargs)
    return snr

#----------------------------------------------------------------------------------------------
# multiprocessing
# can't apply this to optimization over t0, psi because the optimization function is an imported module 

from functools import partial

def one_variable_filter_mp(ranges, npts, finish_fun, 
                        a, A, c, network, freqs, geocent, data, coord, keys, phi, coordinates): 
    """Optimization function, but with azimuth, pole as a single coordinate tuple"""
    az, po = coordinates
    full_func = brute_max(filter_3, ranges, npts, finish_fun, a, A, c, 
                               network, freqs, geocent, data, az, po, coord, keys, 0)[1]
    
    return full_func


def one_variable_mp(*args):
    """Returns function one_variable_filter_mp with just one argument (coordinates), for use in map."""
    one_variable = partial(one_variable_filter_mp, *args)
    
    return one_variable
