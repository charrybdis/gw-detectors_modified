from .geographic_functions import *
import numpy as np

c = 299792458

class Location:
    def __init__(self, name, location, coordinate_system): 
        self.name = name
        self.coordinate_system = coordinate_system
        self.location = location
        
        assert isinstance(location, tuple) or isinstance(location, np.ndarray), 'location must be a tuple or ndarray'
        assert len(list(location)) == 3, 'location must contain 3 coordinates'
        assert coordinate_system == 'ECEF' or coordinate_system == 'geodetic', 'coordinate system must be ECEF or geodetic'

    def convert(self): 
        """Converts location to geodetic if coordinate system is ECEF, and vice versa."""
        if self.coordinate_system == 'ECEF':
            return ECEF_to_geodetic(*location)
        elif self.coordinate_system == 'geodetic':
            return geodetic_to_ECEF(*location)
        else: 
            raise ValueError('coordinate system not ECEF or geodetic')

    def arm_directions(self, arm, divide=True): 
        """Given an arm in ENU local coordinates, will return arms in ECEF."""
        
        if self.coordinate_system == 'ECEF': 
            geodetic_origin = self.convert
            ECEF_poi = ENU_to_ECEF(arm, geodetic_origin)
            arm_direction = ECEF_poi - self.location
        else: 
            geodetic_origin = self.location
            ECEF_poi = ENU_to_ECEF(arm, geodetic_origin)
            arm_direction = ECEF_poi - self.convert
        
        if divide==True: 
            return np.array(arm_direction) / c
        else:
            return arm_direction
        
    def to_array(self, location):
        """Converts a tuple to an array."""
        return np.array(list(location))
    
    def to_lightseconds(self, vector):
        """Divides a vector (location or arms) by the speed of light. Returns an array."""
        if isinstance(vector, np.ndarray):
            c_vector = vector / c
        elif isinstance(vector, tuple): 
            c_vector = to_array(vector) / c
        else:
            raise TypeError('vector must be a numpy array or a tuple')
        return c_vector
        
    @staticmethod
    def to_radians(geodetic_):
        lat, lon, h = geodetic_
        new_tuple = (np.radians(lat), np.radians(lon), h)
        return np.array(new_tuple)
    
    @staticmethod
    def to_degrees(geodetic_):
        lat, lon, h = geodetic_
        new_tuple = (np.degrees(lat), np.degrees(lon), h)
        return np.array(new_tuple)
