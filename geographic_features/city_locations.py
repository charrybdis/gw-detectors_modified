from .locations import *
from .geographic_functions import * 

# name, (lat, lon, h) in degrees
cities = {'Plymouth':(50, -4, 10), 'Ulaanbaatar':(47, 106, 1350), 'Cape Town':(-33, 18, 17),
         'Campo Grande':(-20, -54, 592), 'Perth':(-31, 115, 2), 'Alert':(82, -62, 30)}

ECEF_CITIES = dict()
GEODETIC_CITIES = dict()
for key, value in cities.items(): 
    ECEF_CITIES[key] = Location(key, geodetic_to_ECEF(*Location.to_radians(value)), coordinate_system='ECEF')
    GEODETIC_CITIES[key] = Location(key, Location.to_radians(value), coordinate_system='geodetic')