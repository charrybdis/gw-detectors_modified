import numpy as np
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
    (X, Y, Z) --- (Array)
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
    
    return np.array([X, Y, Z])


def ECEF_to_geodetic(x, y, z): 
    """
    From {https://gis.stackexchange.com/questions/265909/converting-from-ecef-to-geodetic-coordinates}.
    Converts (X, Y, Z) Earth-fixed frame coordinates to (lat, lon, h) geodetic coordinates. 
    
    Parameters: 
    X, Y, Z --- (Number, Number, Number)
    
    Returns:
    (lat, lon, h) --- (Array) latitude, longitude, height/altitude in radians
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

    return np.array([latitude, longitude, altitude]) # in radians


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
    