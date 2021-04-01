"""a module that houses the definition of GW detector objects and networks thereof
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

try:
    from lal.lal import GreenwichMeanSiderealTime as GMST
except:
    GMST = None

#-------------------------------------------------

DEFAULT_COORD = 'celestial'

#-------------------------------------------------

### Power Spectral Density (PSD) objects

class OneSidedPowerSpectralDensity(object):
    """A representation of a power spectral density
    """

    def __init__(self, freqs, psd):
        assert np.all(freqs >= 0), 'frequencies must be positive semi-definite for one-sided power spectral densities'
        self._freqs = freqs

        assert len(freqs)==len(psd), 'frequencies and psd must have the same length'
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

    def draw_noise(self, freqs):
        amp = np.random.randn(*freqs.shape) * (self(freqs)**0.5) ### FIXME: should really be a chi-squared distrib with 2 degrees of freedom?
        phs = np.random.rand(*freqs.shape)*2*np.pi
        return amp*np.cos(phs) + 1j*amp*np.sin(phs)

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
        assert detector.name not in self.names, 'detector named %s already exists in the network!'%detector.name
        self._detectors.append(detector)

    def __iter__(self):
        return self.detectors

    def __len__(self):
        return len(self)

    def snr(self, *args, **kwargs):
        snr = 0.
        for detector in self:
            snr += detector.snr(*args, **kwargs)**2
        return snr**0.5        

#-------------------------------------------------

### Detector objects
# NOTE:
# we only support 2 polarizations in what follows, but we might as well generalize this to support all 6 possible polarizations

class Detector(object):
    """A representation of a ground-based Gravitational Wave detector
    """

    def __init__(self, name, psd, location, long_wavelength_approximation=True, *arms):
        self._name = name
        self.psd = psd
        self.location = np.array(location) ### light-seconds relative to geocenter
        self._long_wavelength_approximation = long_wavelength_approximation

        ### record arms
        self._arms = []
        for arm in arms:
            assert np.shape(arm)==(3,), 'arms must be specified as 3-vectors!'
            self._arms.append(np.array(arm)) ### arm interpreted as the 3 vector defining direction with norm == the light-seconds corresponding to length

    @property
    def name(self):
        return self._name

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, new):
        assert isinstance(new, OneSidedPowerSpectralDensity), 'new PSD must be an instance of OneSidedPowerSpectralDensity'
        self._psd = new

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, new):
        assert np.shape(location) == (3,), 'bad shape for location'
        self._location = np.array(location, dtype=float)

    @property
    def arms(self):
        return self._arms

    @property
    def arm_directions(self):
        return [arm/np.sum(arm**2)**0.5 for arm in self.arms]

    #---

    def draw_noise(self, freqs):
        """simulate Gaussian noise from this detector's PSD
        """
        return self.psd.draw_noise(freqs)

    #---

    def response(self, freqs, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        """\
coord=celestial --> interpret (azimuth, pole) as (RA, Dec) celestial coordinates
coord=geographic --> interpret (azimuth, pole) as (phi, theta) in Earth-fixed coordinates
        """
        if coord == 'geographic':
            phi = azimuth
            theta = pole
        elif coord == 'celestial':
            if GMST == None:
                raise ImportError('could not import lal.lal.GreenwichMeanSiderealTime')
            phi = (azimuth - GMST(geocent_time))%(2*np.pi)
            theta = 0.5*np.pi - pole
        else:
            raise ValueError('coord=%s is not understood!'%coord)

        unphased_response = self.__geographic_response(freqs, phi, theta, psi, long_wavelength_approximation=self.long_wavelength_approximation, *self.arms)
        return unphased_response * self._phase(freqs, geocent_time, phi, theta)

    @staticmethod
    def __geographic_unphased_response(freqs, phi, theta, psi, long_wavelength_approximation=False, *arms):
        """the detector response when angles defining direction to source (phi, theta) are provided in Earth-fixed coordinates \
        NOTE: Child classes should overwrite this!
        """
        raise NotImplementedError('Child classes should overwrite this depending on the number of arms!')

    def _phase(self, freqs, geocent_time, phi, theta):
        """compute the phase shift relative to geocenter. Assumes angles are specified in geographic coordinates (pointing towards the source from geocenter)
        """
        return -2*j*np.pi * freqs * self._dt(phi, theta)

    def _dt(self, phi, theta):
        """
        time delay relative to geocenter
        """
        sinTheta = np.sin(theta)
        n = -np.array([np.cos(phi)*sinTheta, np.sin(phi)*sinTheta, np.cos(theta)]) ### direction of propogation
        return np.sum(self.location*n)

    def project(self, freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        Fp, Fx = self.response(freqs, geocent_time, azimuth, pole, psi, coord=coord)
        return hp*Fp + hx*Fx

    #---

    def snr(self, freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        h = self.project(freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=coord)
        return self._inner_product(freqs, h, h)**0.5

    def _inner_product(self, freqs, a, b):
        return 4*np.trapz(np.conjugate(a)*b/self.psd(freqs), x=freqs)

    def loglikelihood(self, freqs, data, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        h = self.project(freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=coord)
        return -0.5*self._inner_product(freqs, data-h, data-h)

    def filter(self, freqs, data, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        h = self.project(freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=coord)
        h /= self._inner_product(freqs, h, h)**0.5
        return self._inner_product(freqs, data, h)
