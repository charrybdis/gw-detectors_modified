"""a module housing a few known detectors
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np
from pkg_resources import resource_filename

from gwdetectors.detector import OneSidedPowerSpectralDensity

#-------------------------------------------------

def path2psd(path, verbose=False, name=None):
    if verbose:
        print('loading PSD from: '+path)

    if path.endswith('.dat') or path.endswith('.dat.gz') or path.endswith('.txt') or path.endswith('.txt.gz'):
        ans = np.genfromtxt(path, names=True)
        freqs = ans['frequency']
        psd = ans['psd']

    elif path.endswith('.csv') or path.endswith('.csv.gz'):
        ans = np.genfromtxt(path, names=True, delimiter=',')
        freqs = ans['frequency']
        psd = ans['psd']

    else:
        raise ValueError('file format for path=%s not understood!'%path)

    if name is None:
        name = path

    return OneSidedPowerSpectralDensity(freqs, psd, name=name)

#-------------------------------------------------

### Power Spectral Densities

PSDS = dict((name, path2psd(resource_filename('gwdetectors.cache', name+".csv.gz"), name=name)) for name in \
    ['aligo-design', 'aplus-design', 'advirgo-design', 'ce-design'])

#-------------------------------------------------

KNOWN_PSDS = sorted(PSDS.keys())
