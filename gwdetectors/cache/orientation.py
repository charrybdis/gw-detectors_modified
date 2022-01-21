"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

c = 299792458.0 # m/s

#-------------------------------------------------

DETECTOR_ORIENTATIONS = dict()

### known detector locations and orientations

# based on https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html

DETECTOR_ORIENTATIONS['H'] = ( ### LHO
    np.array((-2.16141492636e+06, -3.83469517889e+06, +4.60035022664e+06)) /c, ### location in sec
    (np.array((-0.22389266154, +0.79983062746, +0.55690487831)) * 4e3/c, ### xarm in sec
     np.array((-0.91397818574, +0.02609403989, -0.40492342125)) * 4e3/c, ### yarm in sec
    ),
)

DETECTOR_ORIENTATIONS['L'] = ( ### LLO
    np.array((-7.42760447238e+04, -5.49628371971e+06, +3.22425701744e+06)) / c, # location in sec
    (np.array((-0.95457412153, -0.14158077340, -0.26218911324)) * 4e3/c, ### xarm in sec
     np.array((+0.29774156894, -0.48791033647, -0.82054461286)) * 4e3/c, ### yarm in sec
    ),
)

DETECTOR_ORIENTATIONS['V'] = ( ### Virgo
    np.array((+4.54637409900e+06, +8.42989697626e+05, +4.37857696241e+06)) / c, # location in sec
    (np.array((-0.70045821479, +0.20848948619, +0.68256166277)) * 3e3/c, ### xarm in sec
     np.array((-0.05379255368, -0.96908180549, +0.24080451708)) * 3e3/c, ### yarm in sec
    ),
)

DETECTOR_ORIENTATIONS['K'] = ( ### KAGRA
    np.array((-3777336.024, 3484898.411, 3765313.697)) / c, # location in sec
    (np.array((-0.3759040, -0.8361583,  +0.3994189)) * 3e3/c, ### xarm in sec
     np.array((+0.7164378, +0.01114076, +0.6975620)) * 3e3/c, ### yarm in sec
    ), ### xarm, yarm in sec
)

DETECTOR_ORIENTATIONS['G'] = ( ### GEO 600
    np.array((+3.85630994926e+06, +6.66598956317e+05, +5.01964141725e+06)) / c, ### location in sec
    (np.array((-0.44530676905, +0.86651354130, +0.22551311312)) * 6e2/c, ### xarm in sec
     np.array((-0.62605756776, -0.55218609524, +0.55058372486)) * 6e2/c, ### yarm in sec
    ),
)

#------------------------

### speculative/hypothetical detector locations and orientations

DETECTOR_ORIENTATIONS['CE@H'] = ( ### Cosmic Explorer at LHO
    DETECTOR_ORIENTATIONS['H'][0],
    (DETECTOR_ORIENTATIONS['H'][1][0] * 10, ### arms are 10x longer
     DETECTOR_ORIENTATIONS['H'][1][1] * 10,
    ),
)

DETECTOR_ORIENTATIONS['CE@L'] = ( ### Cosmic Explorer at LLO
    DETECTOR_ORIENTATIONS['L'][0],
    (DETECTOR_ORIENTATIONS['L'][1][0] * 10, ### arms are 10x longer
     DETECTOR_ORIENTATIONS['L'][1][1] * 10,
    ),
)

#-------------------------------------------------

KNOWN_DETECTOR_ORIENTATIONS = sorted(DETECTOR_ORIENTATIONS.keys())
