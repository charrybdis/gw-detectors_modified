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

    def __init__(self, name, psd, location, long_wavelength_approximation=True, *arms):
        self._name = name
        self.psd = psd
        self.location = np.array(location)
        self._long_wavelength_approximation = long_wavelength_approximation

        self._init_arms(*arms)

    def _init_arms(self, *arms): 
        raise NotImplementedError('''\
each arm should be a 3 vector. Length corresponds to the length of the arm, direction is orientation relative to the earth
''')

    @property
    def name(self):
        return self._name

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, new):
        assert isinstance(new, PowerSpectralDensity), 'new PSD must be an instance of PowerSpectralDensity'
        self._psd = new

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, new):
        assert np.shape(location) == (3,), 'bad shape for location'
        self._location = np.array(location, dtype=float)

    #---

    def draw_noise(self, freqs):
        """simulate Gaussian noise from this detector's PSD
        """
        return self.psd.draw_noise(freqs)

    #---

    @staticmethod
    def response(freqs, geocent_time, ra, dec, psi, long_wavelength_approximation=True):
        raise NotImplementedError('''need to convert from ra, dec, psi into the same coordinate system as the detector. Then we can use existing functionality''')

    def project(self, freqs, hp, hx, geocent_time, ra, dec, psi):
        Fp, Fx = self.response(freqs, geocent_time, ra, dec, psi, long_wavelength_approximation=self.long_wavelength_approximation)
        return hp*Fp + hx*Fx

    '''
    def project(self, freqs, hpf, hxf, theta, phi, psi, zeroFreq=False):
        """
        project strains into this detector
        if zeroFreq, we use const_antenna_response instead of antenna_response
        """
        ### angular dependence from antenna patterns
        if zeroFreq:
            Fp, Fx = ant.const_antenna_response(theta, phi, psi, self.ex, self.ey)
        else:
            Fp, Fx = ant.antenna_response(theta, phi, psi, self.ex, self.ey, T=self.T, freqs=freqs)
            Fp = Fp[:,0]
            Fx = Fx[:,0]
        ### overall phase delay from extra time-of-flight
        ### r is measured in seconds
        phs = -twoIpi*freqs*self.dt(theta, phi) ### expect h_IFO(t) = h(t - n*r) = h(t-dt) by my convention

        return (Fp*hpf + Fx*hxf)*np.exp(phs)

    def dt(self, theta, phi):
        """
        time delay relative to geocenter
        """
        sinTheta = np.sin(theta)
        n = -np.array([np.cos(phi)*sinTheta, np.sin(phi)*sinTheta, np.cos(theta)])
        return np.sum(self.r*n)

    def drawNoise(self, freqs):
        """
        simulate Gaussian noise from this detector's PSD
        """
        amp = np.random.randn(*freqs.shape)*self.PSD(freqs)**0.5
        phs = np.random.rand(*freqs.shape)
        return amp*np.cos(phs) + 1j*amp*np.sin(phs)

def h2hAtT(freqs, h, t0):
    """
    add in phasing for time-at-coalescence
    """
    return h*np.exp(twoIpi*freqs*t0)

def h2pol( h, iota, distance=1. ):
    """
    map a waveform into polarization components and normalize by distance
    """
    cos_iota = np.cos(iota)
    return h*0.5*(1+cos_iota**2)/distance, 1j*h*cos_iota/distance
    '''

    #---

    def snr(self, freqs, hp, hx, geocent_time, ra, dec, psi):
        return self._compute_snr(freqs, self.project(freqs, hp, hx, geocent_time, ra, dec, psi))

    def _compute_snr(self, freqs, h):
        raise NotImplementedError('''\
    template = detector.project(freqs, hpf, hxf, theta, phi, psi, zeroFreq=zeroFreq)
    PSD = detector.PSD(freqs)
    deltaF = freqs[1]-freqs[0]

    ans = 2*np.sum(deltaF*np.conjugate(data)*template/PSD).real
    if normalizeTemplate:
        ans /= np.sum(deltaF*np.conjugate(template)*template/PSD).real**0.5

    return ans
''')

class TwoArmDetector(Detector):
    """A representation of a detector with 2 arms
    """

    def __init__(self, name, psd, location, xarm, yarm, long_wavelength_approximation=True):
        Detector.__init__(self, name, psd, location, xarm, yarm, long_wavelength_approximation=long_wavelength_approximation)

    @staticmethod
    def response(freqs, geocent_time, ra, dec, psi, long_wavelength_approximation=True):
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

    def draw_noise(self, freqs):
        amp = np.random.randn(*freqs.shape) * (self(freqs)**0.5) ### FIXME: should really be a chi-squared distrib with 2 degrees of freedom?
        phs = np.random.rand(*freqs.shape)*2*np.pi
        return amp*np.cos(phs) + 1j*amp*np.sin(phs)
