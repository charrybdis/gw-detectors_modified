"""a module that houses 2 arm detector logic

Expressions are taken LIGO-T060237 in a coordinate-dependent form. 
These were then modified to a coordinate independent form by the module's author.

This included validation against Eqn. B7 from Anderson, et all PhysRevD 63(04) 2003
"""
__author__ = "Reed Essick <reed.essick@gmail.com>"

#-------------------------------------------------

import numpy as np

from .detector import (Detector, DEFAULT_COORD)

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
    def _geographic_unphased_response(freqs, phi, theta, psi, arms, long_wavelength_approximation=True):
        '''detector response for 2 arms, where we take the difference between the arms as the signal
        '''
        assert len(arms) == 2, 'must supply exactly 2 arms'
        xarm, yarm = arms
        if long_wavelength_approximation:
            ans = np.ones((2, len(freqs)), dtype=float)
            ans[0,:], ans[1,:] = lwa_antenna_response(phi, theta, psi, xarm, yarm)
            return ans

        else:
            return antenna_response(freqs, phi, theta, psi, xarm, yarm)

#-------------------------------------------------

def lwa_antenna_response(phi, theta, psi, xarm, yarm):
    """\
    computes the antenna patterns for detector arms oriented along xarm and yarm (cartesian 3-vectors) in the long-wavelength approximation
    Antenna patterns are computed accoring to Eqn. B7 from Anderson, et all PhysRevD 63(04) 2003
    """
    # compute angles defining direction to the source frame
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    X = (sin_phi*cos_psi - sin_psi*cos_phi*cos_theta, -cos_phi*cos_psi - sin_psi*sin_phi*cos_theta, sin_psi*sin_theta)
    Y = (-sin_phi*sin_psi - cos_psi*cos_phi*cos_theta, cos_phi*sin_psi - cos_psi*sin_phi*cos_theta, sin_theta*cos_psi)

    ### convert arms into unit vectors
    nx = np.array(xarm)
    nx /= np.sum(nx**2)**0.5

    ny = np.array(yarm)
    ny /= np.sum(ny**2)**0.5

    ### iterate over x,y,z to compute F+ and Fx
    Fp = 0.
    Fx = 0.
    for i in range(3):
        nx_i = nx[i]
        ny_i = ny[i]
        Xi = X[i]
        Yi = Y[i]
        for j in range(3):
            Xj = X[j]
            Yj = Y[j]
            Dij = 0.5*(nx_i*nx[j] - ny_i*ny[j]) ### detector matrix

            Fp += (Xi*Xj - Yi*Yj)*Dij ### add contributions to antenna responses
            Fx += (Xi*Yj + Yi*Xj)*Dij

    return Fp, Fx

#------------------------

def antenna_response(freqs, phi, theta, psi, xarm, yarm, T=1.):
    '''\
    theta, phi, psi, and freqs should all be the same length np.ndarray objects if they are not floats
    xarm and yarm are cartesian 3-vectors defining the directions of the arms. Their norm should be their length in light-seconds.
    '''
    raise NotImplementedError

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
    ex_wave = np.array([sinPhi*cosPsi - sinPsi*cosPhi*cosTheta, -cosPhi*cosPsi - sinPsi*sinPhi*cosTheta, sinPsi*sinTheta])
    ey_wave = np.array([-sinPhi*sinPsi - cosPsi*cosPhi*cosTheta, cosPhi*sinPsi - cosPsi*sinPhi*cosTheta, sinTheta*cosPsi])

    ### compute cartesian vector for the line-of-sight to the source
    n = np.array([sinTheta*cosPhi, sinTheta*sinPhi, cosTheta])

    ### compute detector matrix
    freqsT = 2j*np.pi*freqs*T ### this convention should match what is in LAL : x(f) = \int dt e^{-2\pi i f t} x(t)

    # factor of 1/2 is for normalization
    dV_xx = 0.5 * __D__(freqsT, np.sum(np.outer(ex, np.ones_like(theta))*n, axis=0))
    dV_yy = 0.5 * __D__(freqsT, np.sum(np.outer(ey, np.ones_like(theta))*n, axis=0))

    ### assemble these parts into antenna responses
    Fp = 0.
    Fx = 0.
    for i in range(3): ### FIXME? may be able to do this more efficiently vi np.array manipulations?
        exi = ex[i]
        eyi = ey[i]
        ex_wavei = ex_wave[i]
        ey_wavei = ey_wave[i]
        for j in range(3):
            ex_wavej = ex_wave[j]
            ey_wavej = ey_wave[j]
            Dij = dV_xx * exi*ex[j] - dV_yy * eyi*ey[j] ### compute matrix element for detector matrix

            Fp += Dij * (ex_wavei*ex_wavej - ey_wavei*ey_wavej) ### multiply by matrix element from wave polarizations
            Fx += Dij * (ex_wavei*ey_wavej + ey_wavei*ex_wavej)

    return Fp, Fx


def __D__(freqsT, N):
    '''
    helper function that returns the part of the frequency dependence that depends on the arm's directions
    assumes freqsT = 2j*np.pi*freqs*T
    '''
    if isinstance(freqsT, (int, float, complex)):
        if freqsT==0:
            return np.ones_like(N, dtype=complex)

    elif np.all(freqsT==0):
        return np.ones((len(freqsT), len(N)), dtype=complex)

    n = np.outer(np.ones_like(freqsT), N)
    freqsT = np.outer(freqsT, np.ones_like(N))
    phi = freqsT/1j

    ans = np.empty_like(freqsT)

    truth = n==1
    ans[truth] = 0.5*(1+np.exp(freqsT[truth])*np.sinc(phi[truth]/np.pi))

    truth = n==-1
    ans[truth] = np.exp(-2*freqsT[truth])*0.5*(1 + np.exp(freqsT[truth])*np.sinc(phi[truth]/np.pi))

    truth = freqsT==0
    ans[truth] = 1

    truth = (np.abs(n)!=1)*(freqsT!=0)
    ans[truth] = np.exp(-freqsT[truth])/(1-n[truth]**2)*(np.sinc(phi[truth]/np.pi) + (n[truth]/freqsT[truth])*(np.cos(phi[truth]) - np.exp(freqsT[truth]*n[truth])))

    return ans
