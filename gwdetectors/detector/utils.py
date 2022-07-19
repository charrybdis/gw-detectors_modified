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

    def __init__(self, freqs, psd, name=None):
        assert np.all(freqs >= 0), 'frequencies must be positive semi-definite for one-sided power spectral densities'
        self._freqs = freqs

        assert len(freqs)==len(psd), 'frequencies and psd must have the same length'
        self._psd = psd

        self._name = name

    def __repr__(self):
        freqs = self.freqs
        ans = 'freq within [%.6e %.6e])' % (np.min(freqs), np.max(freqs))
        if self.name is not None:
            ans = self.name + ", " + ans
        return 'OneSidedPowerSpectralDensity(%s)'%ans

    @property
    def name(self):
        return self._name

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
        """\
draw real and imaginary parts of frequency-domain noise from the Whittle likelihood described by this PSD.
Assumes frequency array is regularly spaced
        """
        df = freqs[1] - freqs[0] ### assume frequency array is regularly spaced
        stdv = (self(freqs)/(4*df))**0.5 ### compute the standard deviation of the real, imag noise at each frequency
                                         ### assuming this frequency spacing

        # draw noise from likelihood model
        real, imag = np.random.normal(size=(2, len(freqs))) # Gaussian, independent for real and imag parts
        real *= stdv # scale these by the psd
        imag *= stdv

        # return
        return real + 1j*imag

#-------------------------------------------------

### Detector objects
# NOTE:
# we only support 2 polarizations in what follows, but we might as well generalize this to support all 6 possible polarizations

class Detector(object):
    """A representation of a ground-based Gravitational Wave detector
    """

    def __init__(self, name, psd, location, arms, long_wavelength_approximation=True):
        self._name = name
        self.psd = psd
        self.location = np.array(location) ### light-seconds relative to geocenter
        self.long_wavelength_approximation = long_wavelength_approximation

        ### record arms
        self._arms = []
        for arm in arms:
            assert np.shape(arm)==(3,), 'arms must be specified as 3-vectors!'
            self._arms.append(np.array(arm)) ### arm interpreted as the 3 vector defining direction with norm == the light-seconds corresponding to length

    def __repr__(self):
        ans = type(self).__name__ + '('
        ans += '\n  %s,'%self.name
        ans += '\n  location = (%.6e, %.6e, %.6e) sec,'%tuple(self.location)
        ans += '\n  arms = ('
        for arm in self.arms:
            ans += '\n    (%.6e, %.6e, %.6e) sec,'%tuple(arm)
        ans += '\n  )'
        ans += '\n  psd = %s,'%str(self.psd)
        ans += '\n  long-wavelength approximation = %s,'%str(self.long_wavelength_approximation)
        ans += '\n)'
        return ans  

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
        assert np.shape(new) == (3,), 'bad shape for location'
        self._location = np.array(new, dtype=float)

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

    @staticmethod
    def _geographic_angles(geocent_time, azimuth, pole, coord=DEFAULT_COORD):
        """return geographic coordinates
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

        return phi, theta

    def response(self, freqs, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        """\
coord=celestial --> interpret (azimuth, pole) as (RA, Dec) celestial coordinates
coord=geographic --> interpret (azimuth, pole) as (phi, theta) in Earth-fixed coordinates
        """
        phi, theta = self._geographic_angles(geocent_time, azimuth, pole, coord=coord)

        unphased_response = self._geographic_unphased_response(
            freqs,
            phi,
            theta,
            psi,
            self.arms,
            long_wavelength_approximation=self.long_wavelength_approximation,
        )
        return unphased_response * self._geographic_phase(freqs, phi, theta)

    @staticmethod
    def _geographic_unphased_response(freqs, phi, theta, psi, arms, long_wavelength_approximation=False):
        """the detector response when angles defining direction to source (phi, theta) are provided in Earth-fixed coordinates \
        assumes phi, theta, psi have the same shape
        returns an array with shape : (len(freqs), len(phi))
        NOTE: Child classes should overwrite this!
        """
        raise NotImplementedError('Child classes should overwrite this depending on the number of arms!')

    def phase(self, freqs, geocent_time, azimuth, pole, coord=DEFAULT_COORD):
        phi, theta = self._geographic_angles(geocent_time, azimuth, pole, coord=coord)
        return self._geographic_phase(freqs, phi, theta)

    def _geographic_phase(self, freqs, phi, theta):
        """compute the phase shift relative to geocenter. Assumes angles are specified in geographic coordinates (pointing towards the source from geocenter)
        assumes phi, theta, psi have the same shape
        returns an array with shape : (len(freqs), len(phi))
        """
        return np.exp(-2j*np.pi * np.outer(freqs, self._geographic_dt(phi, theta)))

    def dt(self, geocent_time, azimuth, pole, coord=DEFAULT_COORD):
        """return the delay relative to geocenter
        """
        return self._geographic_dt(*self._geographic_angles(geocent_time, azimuth, pole, coord=coord))

    def _geographic_dt(self, phi, theta):
        """time delay relative to geocenter given input in geographic coordinates
        """
        sinTheta = np.sin(theta)
        n = -np.array([np.cos(phi)*sinTheta, np.sin(phi)*sinTheta, np.cos(theta)]) ### direction of propogation
        return self.location[0]*n[0] + self.location[1]*n[1] + self.location[2]*n[2]

    def project(self, freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        """compute the detector response in the frequency domain and project astrophysical signals into readout \
        assumes:
            freqs, hp, hx are vectors with the same length
            geocent_time, azimuth, pole, psi are scalars
            coord is either "celestial" or "geographic"
        """
        Fp, Fx = self.response(freqs, geocent_time, azimuth, pole, psi, coord=coord)
        return hp*Fp[:,0] + hx*Fx[:,0] ### self.response returns shape (len(freq), len(azimuth))

    #---

    def snr(self, freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        h = self.project(freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=coord)
        return self._inner_product(freqs, h, h).real**0.5

    def _inner_product(self, freqs, a, b):
        return 4*np.trapz(np.conjugate(a)*b/self.psd(freqs), x=freqs)

    def loglikelihood(self, freqs, data, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        h = self.project(freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=coord)
        return -0.5*self._inner_product(freqs, data-h, data-h).real

    def filter(self, freqs, data, hp, hx, geocent_time, azimuth, pole, psi, coord=DEFAULT_COORD):
        h = self.project(freqs, hp, hx, geocent_time, azimuth, pole, psi, coord=coord)
        h /= self._inner_product(freqs, h, h).real**0.5
        return self._inner_product(freqs, data, h)

#-------------------------------------------------

### Network of detectors

class Network(object):
    """A reperesentation of a network of ground-based Gravitational Wave detectors
    """

    def __init__(self, *detectors):
        self._detectors = []
        self.extend(detectors)

    def __repr__(self):
        ans = 'Network('
        for det in self.detectors:
            ans += '\n  %s,'%(' '.join(str(det).split()))
        ans += '\n)'
        return ans

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
        return iter(self.detectors)

    def __len__(self):
        return len(self.detectors)

    def snr(self, *args, **kwargs):
        snr = 0.
        for detector in self:
            snr += detector.snr(*args, **kwargs)**2
        return snr**0.5
