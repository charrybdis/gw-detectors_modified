"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

from .detector import (Detector, TwoArmDetector, PowerSpectralDensity)

#-------------------------------------------------

### Power Spectral Densities

'''
known_psds = dict((name, PSD(psd['freqs'], psd['vals'])) for name, psd in \
    [
        ('aLIGO', psds.aLIGO),
        ('aLIGO_O1', psds.aLIGO_O1),
        ('aLIGO_O2', psds.aLIGO_O2),
        ('aLIGO_O3', psds.aLIGO_O3),
        ('aLIGO_design', psds.aLIGO_design),
        ('aPlus', psds.aPlus),
        ('aPlus_sqzonly', psds.aPlus_sqzonly),
        ('aVirgo', psds.aVirgo),
        ('aVirgo_sqz', psds.aVirgo_sqz),
        ('aVirgo_wb', psds.aVirgo_wb),
        ('CE', psds.CE),
        ('CE_wb', psds.CE_wb),
        ('ET', psds.ET),
        ('Voyager', psds.Voyager),
    ]
)
'''

#-------------------------------------------------

### Detectors

'''
for name in ['aLIGO', 'aLIGO_O1', 'aLIGO_O2', 'aLIGO_O3', 'aLIGO_design', 'aPlus', 'aPlus_sqzonly']:
    known_detectors += [
        Detector(
            name = "L-"+name,
            ex = np.array((-0.9546, -0.1416, -0.2622)),
            ey = np.array((+0.2977, -0.4879, -0.8205)),
            r = np.array((-0.074276, -5.496284, +3.224257))*1e6/c,
            L = 4e3,
            PSD = known_psds[name],
        ),
        Detector(
            name = "H-"+name,
            ex = np.array((-0.2239, +0.7998, +0.5569)),
            ey = np.array((-0.9140, +0.0261, -0.4049)),
            r = np.array((-2.161415, -3.834695, +4.600350))*1e6/c,
            L = 4e3,
            PSD = known_psds[name],
        ),
    ]

for name in ['CE', 'CE_wb', 'Voyager']:
    known_detectors += [
        Detector(
            name = "Llong-"+name,
            ex = np.array((-0.9546, -0.1416, -0.2622)),
            ey = np.array((+0.2977, -0.4879, -0.8205)),
            r = np.array((-0.074276, -5.496284, +3.224257))*1e6/c,
            L = 4e4,
            PSD = known_psds[name],
        ),
        Detector(
            name = "Hlong-"+name,
            ex = np.array((-0.2239, +0.7998, +0.5569)),
            ey = np.array((-0.9140, +0.0261, -0.4049)),
            r = np.array((-2.161415, -3.834695, +4.600350))*1e6/c,
            L = 4e4,
            PSD = known_psds[name],
        ),
    ]

for name in ['aVirgo', 'aVirgo_sqz', 'aVirgo_wb']:
    known_detectors += [
        Detector(
            name = "V-"+name,
            ex = np.array((-0.7005, +0.2085, +0.6826)),
            ey = np.array((-0.0538, -0.9691, +0.2408)),
            r = np.array((+4.546374, +0.842990, +4.378577))*1e6/c,
            L = 3e3,
            PSD = known_psds[name],
        ),
    ]

for name in ['ET']:
    known_detectors += [
        Detector(
            name = "ET1-"+name,
            ex = np.array((-0.70045821479, +0.20848948619, +0.68256166277)),
            ey = np.array((-0.39681482542, -0.73500471881, +0.54982366052)),
            r = np.array((4.54637409900, 0.842989697626, 4.37857696241))*1e6/c,
            L = 1e4,
            PSD = known_psds[name],
        ),
        Detector(
            name = "ET2-"+name,
            ex = np.array((0.30364338937, -0.94349420500, -0.13273800225)),
            ey = np.array((0.70045821479, -0.20848948619, -0.68256166277)),
            r = np.array((4.53936951685, 0.845074592488, 4.38540257904))*1e6/c,
            L = 1e4,
            PSD = known_psds[name],
        ),
        Detector(
            name = "ET3-"+name,
            ex = np.array((+0.39681482542, 0.73500471881, -0.54982366052)),
            ey = np.array((-0.30364338937, 0.94349420500, +0.13273800225)),
            r = np.array((4.54240595075, 0.835639650438, 4.38407519902))*1e6/c,
            L = 1e4,
            PSD = known_psds[name],
        ),
    ]
'''
