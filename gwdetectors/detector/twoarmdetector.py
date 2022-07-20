"""a module that houses 2 arm detector logic

Expressions are taken LIGO-T060237 in a coordinate-dependent form. 
These were then modified to a coordinate independent form by the module's author.

This included validation against Eqn. B7 from Anderson, et all PhysRevD 63(04) 2003
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

from .utils import (Detector, DEFAULT_COORD)

#-------------------------------------------------

class TwoArmDetector(Detector):
    """A representation of a detector with 2 arms
    """

    def __init__(self, name, psd, location, arms, long_wavelength_approximation=True):
        """signature requres exactly 2 arms
        """
        assert len(arms) == 2, 'must supply exactly 2 arms'
        Detector.__init__(self, name, psd, location, arms, long_wavelength_approximation=long_wavelength_approximation)

    #---

    def zenith(self, coord=DEFAULT_COORD):
        phi, theta = self._geographic_zenith()
        if coord=='geographic':
            return phi, theta
        elif coord=='celestial':
            if GMST == None:
                raise ImportError('could not import lal.lal.GreenwichMeanSiderealTime')
            return (phi + GMST(geocent_time))%(2*np.pi), 0.5*np.pi - theta
        else:
            raise ValueError('coord=%s not understood!'%coord)

    def _geographic_zenith(self):
        nx, ny = self.arm_directions

        ### take cross product by hand
        x = +nx[1]*ny[2] - nx[2]*ny[1]
        y = -nx[0]*ny[2] + nx[2]*ny[0]
        z = +nx[0]*ny[1] - nx[1]*ny[0]

        return np.arctan2(y, z), np.arccos(z) ### phi, theta

    #---

    @staticmethod
    def _geographic_unphased_response(freqs, phi, theta, psi, xarm_yarm, long_wavelength_approximation=True):
        '''detector response for 2 arms, where we take the difference between the arms as the signal
        theta, phi, psi, and freqs should all be the same length np.ndarray objects if they are not floats
        xarm_yarm = (xarm, yarm) are cartesian 3-vectors defining the directions of the arms. Their norm should be their length in light-seconds.

        frequency-dependent response is based on the expressions within https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.084004
        polarization tensors are based on expressions within https://journals.aps.org/prd/abstract/10.1103/PhysRevD.91.082002
            
        assumes phi, theta, psi have the same shape
        returns an array with shape : (len(freqs), len(phi))
        '''
        assert len(xarm_yarm) == 2, 'must supply exactly 2 arms'
        xarm, yarm = xarm_yarm

        if isinstance(theta, (int, float)):
            theta = [theta]
            phi = [phi]
            psi = [psi]

        ### compute the trigonometric functions only once
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        cosPhi = np.cos(phi)
        sinPhi = np.sin(phi)
        cosPsi = np.cos(psi)
        sinPsi = np.sin(psi)

        ### compute unit vectors in Earth-Fixed frame from the wave-frame
        ex_wave = np.array([
            sinPhi*cosPsi - sinPsi*cosPhi*cosTheta,
            -cosPhi*cosPsi - sinPsi*sinPhi*cosTheta,
            sinPsi*sinTheta,
        ])

        ey_wave = np.array([
            -sinPhi*sinPsi - cosPsi*cosPhi*cosTheta,
            cosPhi*sinPsi - cosPsi*sinPhi*cosTheta,
            sinTheta*cosPsi,
        ])

        ez_wave = np.array([ # take cross product by hand : Z = X cross Y
            ex_wave[1]*ey_wave[2] - ex_wave[2]*ey_wave[1],
            ex_wave[2]*ey_wave[0] - ex_wave[0]*ey_wave[2],
            ex_wave[0]*ey_wave[1] - ex_wave[1]*ey_wave[0],
        ])

        ### compute cartesian vector for the line-of-sight to the source
        n = np.array([sinTheta*cosPhi, sinTheta*sinPhi, cosTheta])
        n = np.transpose(n)

        ### compute detector matrix

        # convert xarm, yarm to what we need
        Tx = np.sum(xarm**2)**0.5
        ex = xarm/Tx

        Ty = np.sum(yarm**2)**0.5
        ey = yarm/Ty

        if long_wavelength_approximation: # compute the detector matrix at zero frequency
            phsx = phsy = np.zeros_like(freqs, dtype=float)

        else:
            # freqsT = 2j*np.pi*freqs*T ### this convention should match what is in LAL : x(f) = \int dt e^{-2\pi i f t} x(t)
            phsx = 2j*np.pi*freqs*Tx
            phsy = 2j*np.pi*freqs*Ty

        Dxx = TwoArmDetector.n2D(phsx, np.dot(n, ex)) ### returns an array with shape : len(freqs), len(theta)
        Dyy = TwoArmDetector.n2D(phsy, np.dot(n, ey))

        ### assemble these parts into antenna responses
        F_plus = 0.0
        F_cross = 0.0
        F_vecx = 0.0
        F_vecy = 0.0
        F_breath = 0.0
        F_long = 0.0

        for i in range(3):
            exi = ex[i]
            eyi = ey[i]
            ex_wavei = ex_wave[i]
            ey_wavei = ey_wave[i]
            ez_wavei = ez_wave[i]

            for j in range(3):
                ex_wavej = ex_wave[j]
                ey_wavej = ey_wave[j]
                ez_wavej = ez_wave[j]

                ### compute matrix element for detector matrix
                Dij = Dxx*exi*ex[j] - Dyy*eyi*ey[j]

                ### multiply by matrix element from wave polarizations

                # tensor polarizations
                F_plus += Dij * (ex_wavei*ex_wavej - ey_wavei*ey_wavej)
                F_cross += Dij * (ex_wavei*ey_wavej + ey_wavei*ex_wavej)

                # vector polarizations
                F_vecx += Dij * (ex_wavei*ez_wavej + ez_wavei*ex_wavej)
                F_vecy += Dij * (ey_wavei*ez_wavej + ez_wavei*ey_wavej)

                # scalar polarizations
                F_breath += Dij * (ex_wavei*ex_wavej + ey_wavei*ey_wavej)
                F_long += Dij * ez_wavei*ez_wavej # normalization included outside loop

        # add normalization for F_long
        F_long /= 2**0.5

        return F_plus, F_cross, F_vecx, F_vecy, F_breath, F_long

    @staticmethod
    def n2D(freqsT, N):
        '''
        helper function that returns the part of the frequency dependence that depends on the arm's directions
        assumes freqsT = 2j*np.pi*freqs*T

        based on Eq. 4 in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.084004
        '''
        if isinstance(freqsT, (int, float, complex)):
            if freqsT==0:
                return 0.5*np.ones_like(N, dtype=complex)

        elif np.all(freqsT==0):
            return 0.5*np.ones((len(freqsT), len(N)), dtype=complex)

        n = np.outer(np.ones_like(freqsT), N)
        freqsT = np.outer(freqsT, np.ones_like(N))

        ans = np.zeros_like(freqsT, dtype=complex)

        # handle f=0 as special case
        truth = freqsT == 0
        ans[truth] = 0.5

        # handle f!=0 for the rest
        truth = np.logical_not(truth)

        # add the first term
        ans[truth] += np.where(n[truth]==1, freqsT[truth], (1 - np.exp(-freqsT[truth]*(1 - n[truth]))) / (1 - n[truth]))

        # add the second term
        ans[truth] += -np.exp(-2*freqsT[truth]) * np.where(n[truth]==-1, -freqsT[truth], (1 - np.exp(freqsT[truth]*(1 + n[truth])))/(1 + n[truth]))

        # multiply by the prefactor
        ans[truth] /= 4*freqsT[truth]

        return ans
