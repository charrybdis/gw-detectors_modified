"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

from gwdetector.detector import OneSidedPowerSpectralDensity

#-------------------------------------------------

### list known reference PSDs...

"""
list known (reference) PSDs
write a factory to put stuff together?

"""


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
