"""a module that houses the definition of GW detector objects and networks thereof
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

### Network of detectors

class Network(object):
    """A reperesentation of a network of ground-based Gravitational Wave detectors
    """

    def __init__(self, *detectors):
        self._detectors = []
        self.extend(detectors)

    @property
    def detectors(self):
        return self._detectors

    @property
    def names(self):
        return [detector.name for detector in self.detectors]

    def extend(self, detectors):
        for detector in detectors:
            self.append(detector)

    def append(self, detector):
        assert isinstance(detector, Detector), 'detectors must be instances of Detector objects'
        self._detectors.append(detector)

    def __iter__(self):
        return self.detectors

    def __len__(self):
        return len(self)

#-------------------------------------------------

### Detector objects
# NOTE:
# we only support 2 polarizations in what follows, but we might as well generalize this to support all 6 possible polarizations

class Detector(object):
    """A representation of a ground-based Gravitational Wave detector
    """

    def __init__(self, name, psd, long_wavelength_approximation=True, *args, **kwargs):
        self._name = name
        self._psd = psd
        self._long_wavelength_approximation = long_wavelength_approximation

        raise NotImplementedError('''need to pass the arms, etc''')

    @property
    def name(self):
        return self._name

    @property
    def psd(self):
        return self._psd

    #---

    @staticmethod
    def response(freqs, long_wavelength_approximation=True):
        raise NotImplementedError

    def project(self, freqs, hp, hx, time, ra, dec, psi):
        Fp, Fx = self.response(freqs, long_wavelength_approximation=self.long_wavelength_approximation)
        return hp*Fp + hx*Fx

    #---

    def snr(self, freqs, hp, hx, time, ra, dec, psi):
        return self._compute_snr(freqs, self.project(freqs, hp, hx, time, ra, dec, psi))

    def _compute_snr(self, freqs, h):
        raise NotImplementedError

class TwoArmDetector(Detector):
    """A representation of a detector with 2 arms
    """

    @staticmethod
    def response(freqs, long_wavelength_approximation=True):
        raise NotImplementedError

#-------------------------------------------------

### Power Spectral Density (PSD) objects

class PowerSpectralDensity(object):
    """A representation of a power spectral density
    """

    def __init__(self, freqs, psd):
        self._freqs = freqs
        self._psd = psd

    @property
    def freqs(self):
        return self._freqs

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, new):
        assert len(new) == len(self._psd), 'new PSD data must be the same length as existing PSD data'
        self._psd = new

    def __call__(self, freqs):
        return np.interp(freqs, self.freqs, self.psd)
