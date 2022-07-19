"""a module that parses a detector network out of a config file
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

try:
    from ConfigParser import ConfigParser ### Python 2
except ImportError:
    from configparser import ConfigParser ### Python 3

from .detector import (Network, Detector, TwoArmDetector)
from .cache import (KNOWN_DETECTORS, DETECTORS, KNOWN_DETECTOR_ORIENTATIONS, DETECTOR_ORIENTATIONS, KNOWN_PSDS, PSDS)
from .cache import path2psd

#-------------------------------------------------

def parse_psd(config, section, verbose=False):
    """parse the PSD from a config
    """
    assert config.has_option(section, 'psd'), 'Config section=%s must have option "psd"'%section
    psd = config.get(section, 'psd')
    if psd in KNOWN_PSDS:
        if verbose:
            print('using known PSD: '+psd)
        psd = PSDS[psd]

    else:
        psd = path2psd(psd, verbose=verbose)

    return psd

def parse_long_wavelength_approximation(config, section, verbose=True):
    """parse the long wavelength approximation flag from config
    """
    if config.has_option(section, 'long_wavelength_approximation'):
        lwa = config.getboolean(section, 'long_wavelength_approximation')
    else:
        if verbose:
            print('defaulting to long_wavelength_approximation=True')
        lwa = True

    return lwa

def parse_vector(config, section, option, verbose=True):
    """parse 3-vectors from config
    """
    if verbose:
        print('parsing 3-vector for : '+option)
    vec = np.array([float(_) for _ in config.get(section, option).split()], dtype=float)
    assert len(vec) == 3, 'location must be a vector with 3 elements'
    return vec

def parse_config_section(section, config, verbose=False):
    """advanced parsing logic to get a detector out of a config section
    """
    if section in KNOWN_DETECTORS: ### just load this without looking for a specific PSD
        if verbose:
            print('using known detector: '+section)
        return DETECTORS[section]

    elif section in KNOWN_DETECTOR_ORIENTATIONS: ### use this orientation
        if verbose:
            print('using known detector orientation: '+section)

        # look up known location, arms
        instantiator, location, arms = DETECTOR_ORIENTATIONS[section]

        # load psd
        psd = parse_psd(config, section, verbose=verbose)

        # load long wavelength approximation
        lwa = parse_long_wavelength_approximation(config, section, verbose=verbose)

        # instantiate and return
        return instantiator(section, psd, location, arms, long_wavelength_approximation=lwa)

    else: ### parse orientation at the same time as the PSD, etc
        if verbose:
            print('parsing detector from : '+section)

        # figure out the detector type
        assert config.has_option(section, 'detector_type'), \
            'Config section=%s must have option "detector_type"'%section
        detector_type = config.get(section, 'detector_type')
        if verbose:
            print('detector_type : '+detector_type)

        # parse location
        assert config.has_option(section, 'location'), \
            'Config section=%s must have option "location"'%section
        location = parse_vector(config, section, 'location', verbose=verbose)

        # parse arms, instantiator
        if detector_type == 'TwoArmDetector':
            instantiator = TwoArmDetector
            arms = []
            for key in ['xarm', 'yarm']:
                assert config.has_option(section, key), \
                    'Config section=%s must have option "%s"'%(section, key)
                arms.append(parse_vector(config, section, key, verbose=verbose))

        else:
            raise RuntimeError('section=%s : detector_type=%s not understood!'%(section, detector_type))

        # parse psd
        psd = parse_psd(config, section, verbose=verbose)

        # parse long wavelength approximation
        lwa = parse_long_wavelength_approximation(config, section, verbose=verbose)

        # instantiate and return
        return instantiator(section, psd, location, arms, long_wavelength_approximation=lwa)

def parse(path, verbose=False, exit_on_exception=False):
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
        except (RuntimeError, AssertionError) as e:
            if exit_on_exception:
                raise e
            elif verbose:
                print('section=%s not understood; skipping'%name)

    ### make sure we got at least one detector
    assert len(network), 'must have at least one detector in our network!'

    ### return
    return network
