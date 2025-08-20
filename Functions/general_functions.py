import numpy as np
from gwdetectors.detector import *
import warnings

#-------------------------------------------------------------------------------------
def c():
    """
    speed of light in m/s
    """
    return 299792458


def sine_Gaussian(t, a, A, c, dt=0, p=0):
    """
    Sinusoidal Gaussian pulse.
    
    Parameters: 
    w --- (Number or 1D array) independent variable frequency (Hz) 
    a --- (Number) frequency of sinuisoidal component
    A --- (Number) amplitude of Gaussian envelope 
    c --- (Number) standard deviation of Gaussian envelope
    
    dt --- (Number) time shift, default 0 
    p --- (Number) phase shift, default 0
    """
    return A * np.cos((2 * np.pi * a * (t - dt)) + p) * np.exp(-(t - dt)**2 / (2 * c**2))


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
    
    if a-width < 0: 
        freqs = np.linspace(0, a+width, numpts)
        signal = ft_sine_Gaussian(freqs, a, A, c, dt, p)
        warnings.warn('Frequency array below zero') 
    else:
        freqs = np.linspace(a-width, a+width, numpts)
        signal = ft_sine_Gaussian(freqs, a, A, c, dt, p)
    
    return freqs, signal


def az_po_meshgrid(res, coord, truncate=False):
    """
    Returns appropriate arrays of sky coordinate based on the coordinate system for use in meshgrid. 
    
    Parameters: 
    res --- (Int) resolution of the grid
    coord --- (String) either 'geographic' or 'celestial'
    
    Returns: 
    azimuths, poles --- (Tuple of 1D arrays of length true_res) arrays of azimuth, pole coordinates
    """
    if coord =='geographic':
        if truncate == True:
            azimuths = np.linspace(-np.pi, np.pi, res)[1:] # since -pi and pi are the same
            poles = np.flip(np.linspace(0, np.pi, res))[1:-1] # since 0 and pi only need to be calculated once for all azimuths
        else: 
            azimuths = np.linspace(-np.pi, np.pi, res)
            poles = np.flip(np.linspace(0, np.pi, res))
    if coord =='celestial':
        pass # maybe add this later
    
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