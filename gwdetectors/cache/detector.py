"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

from .orientation import DETECTOR_ORIENTATIONS
from .psd import PSDS

#-------------------------------------------------

### Detectors with reference PSDs

NAME_TEMPLATE = "%s_%s"
DETECTORS = dict()

### construct specific combinations of known orientations and PSDs

for orientations, psds in [
        (('L', 'H'), ('aligo-design', 'aplus-design',)), ### LLO and LHO with design sensitivities
        (('V',), ('advirgo-design',)),                   ### Virgo with design sensitivity
        (('CE@L', 'CE@H',), ('ce-design',)),             ### CE at LLO, LHO location (but longer arms) with CE PSD
    ]:
    for orientation in orientations:
        instantiator, loc, arms = DETECTOR_ORIENTATIONS[orientation]
        for psd in psds:
            name = NAME_TEMPLATE%(orientation, psd)
            DETECTORS[name] = instantiator(
                name,
                PSDS[psd],
                loc,
                arms,
                long_wavelength_approximation=False, ### NOTE: always assume the long-wavelength approximation
            )                                        ### is False for the default detectors

#------------------------

KNOWN_DETECTORS = sorted(DETECTORS.keys())
