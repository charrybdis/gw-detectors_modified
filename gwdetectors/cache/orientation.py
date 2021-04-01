"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

from gwdetectors.detector import TwoArmDetector

#-------------------------------------------------

c = 299792458.0 # m/s

#-------------------------------------------------

DETECTOR_ORIENTATIONS = dict()

DETECTOR_ORIENTATIONS['LHO'] = (
    TwoArmDetector,
    np.array((-2.161415, -3.834695, +4.600350), dtype=float) * 1e6/c, ### location in sec
    (np.array((-0.2239, +0.7998, +0.5569)) * 4e3/c, np.array((-0.9140, +0.0261, -0.4049)) * 4e3/c), ### xarm, yarm in sec
)

DETECTOR_ORIENTATIONS['LLO'] = (
    TwoArmDetector,
    np.array((-0.074276, -5.496284, +3.224257)) * 1e6/c, # location in sec
    (np.array((-0.9546, -0.1416, -0.2622)) * 4e3/c, np.array((+0.2977, -0.4879, -0.8205)) * 4e3/c), ### xarm, yarm in sec
)

DETECTOR_ORIENTATIONS['Virgo'] (
    TwoArmDetector,
    np.array((+4.546374, +0.842990, +4.378577)) * 1e6/c, # location in sec
    (np.array((-0.7005, +0.2085, +0.6826)) * 3e3/c, np.array((-0.0538, -0.9691, +0.2408)) * 3e3/c), ### xarm, yarm in sec
)

### FIXME: add Kagra, CE, ET, else?

#------------------------

KNOWN_DETECTOR_ORIENTATIONS = sorted(DETECTOR_ORIENTATIONS.keys())
