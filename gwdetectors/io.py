"""a module that parses a detector network out of a config file
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

try:
    from ConfigParser import ConfigParser ### Python 2
except ImportError:
    from configparser import ConfigParser ### Python 3

from .detector import Network
from .cache import (KNOWN_DETECTORS, DETECTORS, KNOWN_DETECTOR_ORIENTATIONS, DETECTOR_ORIENTATIONS, KNOWN_PSDS, PSDS)

#-------------------------------------------------

def path2psd(path, verbose=False):
    if verbose:
        print('loading PSD from: '+psd)
    raise NotImplementedError('load in the PSD from disk')

#-------------------------------------------------

def parse_config(section, config, verbose=False):
    """advanced parsing logic to get a detector out of a config section
    """
    if section in cache.KNOWN_DETECTORS: ### just load this without looking for a specific PSD
        if verbose:
            print('using known detector: '+section)
        return DETECTORS[section]

    elif section in cache.KNOWN_DETECTOR_ORIENTATIONS:
        if verbose:
            print('using known detector orientation: '+section)
        instantiator, location, xarm, yarm = cache.DETECTOR_ORIENTATIONS

        if config.has_option(section, 'long_wavelength_approximation'):
            long_wavelength_approximation = config.getbool(section, 'long_wavelength_approximation')
        else:
            if verbose:
                print('defaulting to long_wavelength_approximation=True')
            long_wavelength_approximation = True

        psd = config.get_option(section, 'psd')
        if psd in cache.KNOWN_PSDS:
            if verbose:
                print('using known PSD: '+psd)
            psd = cache.PSDS[psd]

        else:
            psd = path2psd(path, verbose=verbose)

        return instantiator(section, psd, location, xarm, yarm, long_wavelength_approximation=long_wavelength_approximation)

    else:
        raise RuntimeError('detector=%s not understood!'%section)

def parse(path, verbose=False):
    """parse a network of known detectors out of a config file
    """
    if verbose:
        print('loading config from: '+path)
    config = ConfigParser()
    config.read(path)

    ### iterate over sections and create detectors
    network = Network()
    for name in config.sections(): ### iterate through config's sections
        if name in cache.KNOWN_DETECTOR_ORIENTATIONS:
            network.append(parse_config_section(name, config, verbose=verbose))
        elif verbose:
            print('section=%s not understood; skipping'%name)

    ### make sure we got at least one detector
    assert len(network), 'must have at least one detector in our network!'

    ### return
    return network
