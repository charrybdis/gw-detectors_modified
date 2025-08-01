from ..Functions.general_functions import *

c = c()

class Location:
    def __init__(self, name, location, coordinate_system): 
        self.name = name
        self.coordinate_system = coordinate_system
        self.location = location
        
        assert len(list(*location)) == 3, 'location must be a tuple of 3 values'
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
        """For arms in ENU local coordinates. Will always return arms in ECEF."""
        
        if self.coordinate_system == 'ECEF': 
            geodetic_origin = self.convert
            ECEF_poi = ENU_to_ECEF(arm, geodetic_origin)
            arm_direction = ECEF_poi - self.location
        else: 
            geodetic_origin = self.location
            ECEF_poi = ENU_to_ECEF(arm, geodetic_origin)
            arm_direction = ECEF_poi - self.convert
        
        if divide==True: 
            return arm_direction / c
        else:
            return arm_direction
        
    @staticmethod
    def to_radians(geodetic_tuple):
        lat, lon, h = geodetic_tuple
        new_tuple = (np.radians(lat), np.radians(lon), h)
        return new_tuple
    
    @staticmethod
    def to_degrees(geodetic_tuple):
        lat, lon, h = geodetic_tuple
        new_tuple = (np.degrees(lat), np.degrees(lon), h)
        return new_tuple
