from .locations import *

# name, (lat, lon, h) in degrees
cities = {'Plymouth':(50, -4, 10), 'Ulaanbaatar':np.array(47, 106, 1350), 'Cape Town':(-33, 18, 17),
         'Campo Grande':(-20, -54, 592), 'Perth':(-31, 115, 2), 'Alert':(82, -62, 30)}

CITIES = dict()
for key, value in cities.items(): 
    CITIES[key] = Location(key, Location.to_radians(value), coordinate_system='geodetic')