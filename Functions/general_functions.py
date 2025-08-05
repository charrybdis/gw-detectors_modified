import numpy as np
from gwdetectors.detector import *
import warnings

#------------------------------------------------------------------------------------------------
# general functions (not optimization)

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
    else:
        freqs= np.linspace(a-width, a+width, numpts)
        ast_signal = ft_sine_Gaussian(freqs, a, A, c, dt, p)
    
    return freqs, ast_signal


def az_po_meshgrid(res, coord, endpoint=True):
    """
    Returns appropriate arrays of sky coordinate based on the coordinate system for use in meshgrid. 
    
    Parameters: 
    res --- (Int) resolution of the grid
    coord --- (String) either 'geographic' or 'celestial'
    
    Returns: 
    azimuths, poles --- (Tuple of 1D arrays of length true_res) arrays of azimuth, pole coordinates
    """
    
    if coord =='geographic':
        azimuths = np.linspace(-np.pi, np.pi, res, endpoint=endpoint) # optionally excludes pi, since it's the same as -pi
        # but note that it will change the grid spacing which could be undesirable
        poles = np.flip(np.linspace(0, np.pi, res))
    if coord =='celestial':
        pass
    
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

#------------------------------------------------------------------------------------------------
# geographic functions (mainly coordinate transforms)

def geodetic_to_ECEF(lat, lon, h): 
    """
    Converts (Latitude, Longitude, Altitude) geodetic coordinates to Earth-fixed frame coordinates (X, Y, Z).
    
    Parameters: 
    lat --- (Number) latitude in radians
    lon --- (Number) longitude in radians
    h --- (Number) height/altitude
    
    Returns: 
    X, Y, Z --- (Number, Number, Number)
    """
    a = 6378137.0 # equatorial radius of earth
    b = 6356752.3142 # polar radius of earth
    
    cosPhi = np.cos(lat)
    sinPhi = np.sin(lat)
    cosLda = np.cos(lon)
    sinLda = np.sin(lon)
    
    N = a ** 2 / np.sqrt((a**2 * cosPhi**2) + (b**2 * sinPhi**2))
    
    X = (N + h)* cosPhi * cosLda
    Y = (N + h) * cosPhi * sinLda
    Z = ((b**2/a**2) * N + h) * sinPhi
    
    return X, Y, Z


def ECEF_to_geodetic(x, y, z): 
    """
    From {https://gis.stackexchange.com/questions/265909/converting-from-ecef-to-geodetic-coordinates}.
    Converts (X, Y, Z) Earth-fixed frame coordinates to (lat, lon, h) geodetic coordinates. 
    
    Parameters: 
    X, Y, Z --- (Number, Number, Number)
    
    Returns: 
    lat --- (Number) latitude in radians
    lon --- (Number) longitude in radians
    h --- (Number) height/altitude
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis in meters
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis in meters
    e_sq = f * (2 - f)  # Eccentricity squared

    # Calculate longitude
    longitude = np.arctan2(y, x)

    # Iterative solution for latitude and height
    p = np.sqrt(x**2 + y**2)
    
    # Initial guess for latitude (using a common approximation)
    latitude = np.arctan2(z, p * (1 - e_sq))

    # Iteration until convergence
    for _ in range(10): # A fixed number of iterations is usually sufficient for high accuracy
        N = a / np.sqrt(1 - e_sq * np.sin(latitude)**2)
        h_new = p / np.cos(latitude) - N
        latitude_new = np.arctan2(z, p * (1 - e_sq * (N / (N + h_new))))
        
        if abs(latitude_new - latitude) < 1e-10: # Check for convergence
            latitude = latitude_new
            break
        latitude = latitude_new
    
    # Final height calculation
    N = a / np.sqrt(1 - e_sq * np.sin(latitude)**2)
    altitude = p / np.cos(latitude) - N

    return latitude, longitude, altitude # in radians


def great_circle(t, p1, p2): 
    """
    Parametric equation for a great circle passing through p1 and p2. p1 and p2 are vectors (should probably use ECEF for this).
    
    Parameters: 
    t --- (Array) variable that parametrizes the great circle
    p1, p2 --- (Array x2) linearly independent vectors (in the center of earth frame)
    """
    
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)
    
    vec3 = np.cross(np.cross(p1_norm, p2_norm), p1_norm) 
    vec3_norm  = np.linalg.norm(vec3)
    
    R_t = np.cos(t)[:, np.newaxis] * p1_norm + np.sin(t)[:, np.newaxis] * vec3_norm
    return R_t


def ENU_to_ECEF(ENU_coords, geodetic_origin): 
    """
    {https://gis.stackexchange.com/questions/308445/local-enu-point-of-interest-to-ecef}
    Converts (E, N, U) coordinates to (X, Y, Z) ECEF coordinates. 
    
    Parameters: 
    ENU_coords --- (Array or Tuple).  
    geodetic_origin --- (Number, Number, Number) local point of origin in geodetic coordinates. 
    
    Returns: 
    ECEF_poi --- (Array) 
    """
    lon, lat, h = geodetic_origin
    
    sinLda = np.sin(lon) 
    cosLda = np.cos(lon)
    sinPhi = np.sin(lat)
    cosPhi = np.cos(lat)
    
    R = np.array([-sinLda, -cosLda * sinPhi, cosLda * cosPhi], 
                [cosLda, -sinLda * sinPhi, sinLda * cosPhi], 
                [0, cosPhi, sinPhi])
    
    ECEF_origin = geodetic_to_ECEF(*geodetic_origin)
    
    ECEF_poi = np.matmul(R, np.array(ENU_coords)) + ECEF_origin
    
    return ECEF_poi
    