"""a module that parses a detector network out of a config file
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

try:
    from ConfigParser import ConfigParser ### Python 2
except ImportError:
    from configparser import ConfigParser ### Python 3

from .detector import (Network)
from .cache import (KNOWN_DETECTORS, DETECTORS, KNOWN_DETECTOR_ORIENTATIONS, DETECTOR_ORIENTATIONS, KNOWN_PSDS, PSDS)
from .cache import path2psd

#-------------------------------------------------

def parse_config_section(section, config, verbose=False):
    """advanced parsing logic to get a detector out of a config section
    """
    if section in KNOWN_DETECTORS: ### just load this without looking for a specific PSD
        if verbose:
            print('using known detector: '+section)
        return DETECTORS[section]

    elif section in KNOWN_DETECTOR_ORIENTATIONS:
        if verbose:
            print('using known detector orientation: '+section)
        instantiator, location, arms = DETECTOR_ORIENTATIONS[section]

        if config.has_option(section, 'long_wavelength_approximation'):
            long_wavelength_approximation = config.getboolean(section, 'long_wavelength_approximation')
        else:
            if verbose:
                print('defaulting to long_wavelength_approximation=True')
            long_wavelength_approximation = True

        assert config.has_option(section, 'psd'), 'Config section=%s must have option "psd"'%section
        psd = config.get(section, 'psd')
        if psd in KNOWN_PSDS:
            if verbose:
                print('using known PSD: '+psd)
            psd = PSDS[psd]

        else:
            psd = path2psd(psd, verbose=verbose)

        return instantiator(section, psd, location, arms, long_wavelength_approximation=long_wavelength_approximation)

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
        try:
            network.append(parse_config_section(name, config, verbose=verbose))
        except RuntimeError:
            if verbose:
                print('section=%s not understood; skipping'%name)

    ### make sure we got at least one detector
    assert len(network), 'must have at least one detector in our network!'

    ### return
    return network
