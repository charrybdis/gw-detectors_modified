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

def inner_product(freqs, psd, a, b):
    return 4*np.trapz(np.conjugate(a)*b/psd, x=freqs)

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

    def _marginal_variances(self, freqs):
        df = freqs[1] - freqs[0]  ### assume frequency array is regularly spaced
        return self(freqs)/(4*df) ### compute the standard deviation of the real, imag noise at each frequency
                                  ### assuming this frequency spacing

    def draw_noise(self, freqs):
        """\
draw real and imaginary parts of frequency-domain noise from the Whittle likelihood described by this PSD.
Assumes frequency array is regularly spaced
        """
        stdv = self._marginal_variances(freqs)**0.5

        # draw noise from likelihood model
        real, imag = np.random.normal(size=(2, len(freqs))) # Gaussian, independent for real and imag parts
        real *= stdv # scale these by the psd
        imag *= stdv

        # return
        return real + 1j*imag

    def logprob(self, freqs, data):
        """the probability of obtaining data as a noise realization.
        Assumes equally spaced frequency samples when computing determinant
        """
        df = freqs[1] - freqs[0] ### assume frequency array is regularly spaced when computing determinant
        return -0.5*inner_product(freqs, self(freqs), data, data).real \
            -0.5*len(freqs)*np.log(2*np.pi) - 0.5*np.sum(np.log(self._marginal_variances(freqs)))

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

    def project(
            self,
            freqs,
            geocent_time,
            azimuth,
            pole,
            psi,
            coord=DEFAULT_COORD,
            hp=None,  ### "plus" tensor mode
            hx=None,  ### "cross" tensor mode
            hvx=None, ### "x" vector mode
            hvy=None, ### "y" vector mode
            hb=None,  ### "breathing" scalar mode
            hl=None,  ### "longitudinal" scalar mode
        ):
        """compute the detector response in the frequency domain and project astrophysical signals into readout \
        assumes:
            freqs, hp, hx are vectors with the same length
            geocent_time, azimuth, pole, psi are scalars
            coord is either "celestial" or "geographic"
        """
        # compute antenna responses
        Fp, Fx, Fvx, Fvy, Fb, Fl = self.response(freqs, geocent_time, azimuth, pole, psi, coord=coord)

        # iterate and add contributions from each polarization that is present
        ans = 0.0
        for h, F in [(hp, Fp), (hx, Fx), (hvx, Fvx), (hvy, Fvy), (hb, Fb), (hl, Fl)]:
            if h is not None:
                ans += h*F[:,0] ### self.response returns shape (len(freq), len(azimuth))
        return ans

    #---

    def snr(self, freqs, strain):
        """computes the optimal SNR. Function takes the same args, kwargs as project()
        """
        return inner_product(freqs, self.psd(freqs), strain, strain).real**0.5

    def logprob(self, freqs, data):
        return self.psd.logprob(freqs, data)

    def filter(self, freqs, data, strain):
        psd = self.psd(freqs)
        return inner_product(freqs, psd, data, strain/inner_product(freqs, psd, strain, strain).real**0.5)

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

    #---

    def project(self, *args, **kwargs):
        return [det.project(*args, **kwargs) for det in self.detectors]

    #---

    def snr(self, freqs, *args, **kwargs):
        return np.sum([det.snr(freqs, det.project(freqs, *args, **kwargs))**2 for det in self.detectors])**0.5

    def logprob(self, freqs, data):
        return np.sum([det.logprob(freqs, d) for det, d in zip(self.detectors, data)], axis=0)

    def filter(self, freqs, data, strain):
        return np.sum([det.filter(freqs, d, strain)**2 for det, d in zip(self.detectors, data)])**0.5
