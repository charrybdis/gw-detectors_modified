"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

from .orientation import *
from .psd import *

#-------------------------------------------------

### Detectors with reference PSDs

NAME_TEMPLATE = "%s_%s"
DETECTORS = dict()

### iterate and construct all possible combinations of known orientations and PSDs
for orientation_name, (instantiator, location, arms) in DETECTOR_ORIENTATIONS.items():
    for psd_name, psd in PSDS.items():
        _ = instantiator(NAME_TEMPLATE%(orientation_name, psd_name), psd, location, *arms)
        detectors[_.name] = _

#------------------------

KNOWN_DETECTORS = DETECTORS.keys()
