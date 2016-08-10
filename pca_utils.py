# -*- coding: utf-8 -*-
"""
Created on June 02 2016

@author: Rafael Neto Henriques (rafaelnh21@gmail.com)
"""

# import relevant modules
import numpy as np
from dipy.sims.voxel import (multi_tensor, _add_gaussian, _add_rician,
                             _add_rayleigh)

def rfiw_phantom(gtab, snr=None, noise_type='rician'):
    """retangle fiber immersed in water"""

    # define voxel index
    slice_ind = np.zeros((10, 10, 10))
    slice_ind[4:7, 4:7, :] = 1
    slice_ind[4:7, 7, :] = 2
    slice_ind[7, 7, :] = 3
    slice_ind[7, 4:7, :] = 4
    slice_ind[7, 3, :] = 5
    slice_ind[4:7, 3, :] = 6
    slice_ind[3, 3, :] = 7
    slice_ind[3, 4:7, :] = 8
    slice_ind[3, 7, :] = 9

    # Define tisse diffusion parameters
    # Restricted diffusion
    ADr = 0.99e-3
    RDr = 0.0
    # Hindered diffusion
    ADh = 2.26e-3
    RDh = 0.87
    # S0 value for tissue
    S1 = 50
    # Fraction between Restricted and Hindered diffusion
    fia = 0.51

    # Define water diffusion
    Dwater = 3e-3
    S2 = 100  # S0 value for water

    # Define tissue volume fraction for each voxel type (in index order)
    f = np.array([0., 1., 0.6, 0.18, 0.30, 0.15, 0.50, 0.35, 0.70, 0.42])

    # Define S0 for each voxel (in index order)
    S0 = S1*f + S2*(1-f)

    # multi tensor simulations assume that each water pull as constant S0
    # since I am assuming that tissue and water voxels have different S0,
    # tissue volume fractions have to be adjusted to the measured f values when
    # constant S0 are assumed constant. Doing this correction, simulations will
    # be analogous to simulates that S0 are different for each media. (For more
    # datails on this contact the phantom designer)
    f1 = f * S1/S0

    mevals = np.array([[ADr, RDr, RDr], [ADh, RDh, RDh],
                       [Dwater, Dwater, Dwater]])
    angles=[(0, 0, 1), (0, 0, 1), (0, 0, 1)]
    DWI = np.zeros((10, 10, 10, gtab.bvals.size))
    for i in range(10):
        fractions = [f1[i]*fia*100, f1[i] * (1-fia) * 100, (1 - f1[i]) * 100]
        sig, direction = multi_tensor(gtab, mevals, S0=S0[i], angles=angles,
                                      fractions=fractions, snr=None)
        DWI[slice_ind == i, :] = sig

    if snr is None:
        return DWI
    else:
        noise_adder = {'gaussian': _add_gaussian,
                       'rician': _add_rician,
                       'rayleigh': _add_rayleigh}
        
        sigma = S2 * 1.0 /snr
        n1 = np.random.normal(0, sigma, size=DWI.shape)
        if noise_type == 'gaussian':
            n2 = None
        else:
            n2 = np.random.normal(0, sigma, size=DWI.shape)

        return noise_adder[noise_type](DWI, n1, n2)

# -----------------------------------------------------------------
# Fiber segments phantom
# -----------------------------------------------------------------

def fiber_segments_phantom(gtab, fiber_sigma, snr=None, noise_type='rician'):
    Phantom = np.zeros((10, 10, 10, gtab.bvals.size))
    n1 = np.random.normal(90, fiber_sigma, size=Phantom.shape[:-1])
    n2 = np.random.normal(0, fiber_sigma, size=Phantom.shape[:-1])

    ADr = 0.99e-3
    RDr = 0.0
    ADh = 2.26e-3
    RDh = 0.87
    S1 = 50
    fia = 0.51

    mevals = np.array([[ADr, RDr, RDr], [ADh, RDh, RDh]])
    fractions = [fia*100, (1-fia) * 100]

    for i in range(10):
        for j in range(10):
            for k in range(10):
                angles=[(n1[i, j, k], n2[i, j, k]), (n1[i, j, k], n2[i, j, k])]
                sig, direction = multi_tensor(gtab, mevals, S0=S1,
                                              angles=angles,
                                              fractions=fractions,
                                              snr=None)
                Phantom[i, j, k, :] = sig

    if snr is None:
        return Phantom
    else:
        noise_adder = {'gaussian': _add_gaussian,
                       'rician': _add_rician,
                       'rayleigh': _add_rayleigh}
        
        sigma = S1 * 1.0 /snr
        n1 = np.random.normal(0, sigma, size=Phantom.shape)
        if noise_type == 'gaussian':
            n2 = None
        else:
            n2 = np.random.normal(0, sigma, size=Phantom.shape)

        return noise_adder[noise_type](Phantom, n1, n2)
    

# -----------------------------------------------------------------
# PCA
# -----------------------------------------------------------------

def mp_distribution(x, var, y):
    """ Samples the Marchenko–Pastur probability distribution

    Parameters
    ----------
    x : array (N,)
        Values of random variable to sample the probability distribution
    var : float
        Variance of the random variable
    y : float
        Parameter associated to the matrix X that produces the distributions.
        This X is a M x N random matrix which columns entries are identical
        distributed random variables with mean 0 and given variance, y is given
        by N/M.
    """
    xpos = var * (1 + np.sqrt(y)) ** 2
    xneg = var * (1 - np.sqrt(y)) ** 2

    p = np.zeros(x.shape)
    xdis = np.logical_and(x<xpos, x>xneg)
    p[xdis] = np.sqrt((xpos-x[xdis]) * (x[xdis]-xneg)) / (2*np.pi*var*y*x[xdis])

    return p

def pca_noise_classifier(L, m):
    """ Classify which PCA eigenvalues are related to noise

    Parameters
    ----------
    L : array (n,)
        Array containing the PCA eigenvalues.

    Returns
    -------
    c : int
        Number of eigenvalues related to noise
    sig2 : float
        Estimation of the noise variance
    """
    sig2 = np.mean(L)
    c = L.size - 1
    r = L[c] - L[0] - 4 * np.sqrt((c+1.0) / m) * sig2
    while r > 0:
        sig2 = np.mean(L[:c])
        c = c - 1
        r = L[c] - L[0] - 4*np.sqrt((c+1.0) / m) * sig2
    return c + 1, sig2

def pca_denoising(dwi, ps=2, overcomplete=False):
    """ Denoises DWI volumes using PCA analysis and Marchenko–Pastur
    probability theory

    Parameters
    ----------
    dwi : array ([X, Y, Z, g])
        Matrix containing the 4D DWI data.
    ps : int
        Number of neighbour voxels for the PCA analysis.
        Default: 2
    overcomplete : boolean
        If set to True, overcomplete local PCA is computed
        Default: False

    Returns
    -------
    den : array ([X, Y, Z, g])
        Matrix containing the denoised 4D DWI data.
    std : array ([X, Y, Z])
        Matrix containing the noise std estimated using
        Marchenko-Pastur probability theory.
    ncomps : array ([X, Y, Z])
        Number of eigenvalues preserved for the denoised
        4D data.
    """
    # Compute dimension of neighbour sliding window
    m = (2*ps + 1) ** 3

    n = dwi.shape[3]
    den = np.zeros(dwi.shape)
    ncomps = np.zeros(dwi.shape[:3])
    sig2 = np.zeros(dwi.shape[:3])
    if overcomplete:
        wei = np.zeros(dwi.shape)

    for k in range(ps, dwi.shape[2] - ps):
        for j in range(ps, dwi.shape[1] - ps):
            for i in range(ps, dwi.shape[0] - ps):
                # Compute eigenvalues for sliding window
                X = dwi[i - ps: i + ps + 1, j - ps: j + ps + 1,
                        k - ps: k + ps + 1, :]
                X = X.reshape(m, n)
                M = np.mean(X, axis=0)
                X = X - M
                [L, W] = np.linalg.eigh(np.dot(X.T, X)/m)

                # Find number of noise related eigenvalues
                c, sig = pca_noise_classifier(L, m)

                # Reconstruct signal without noise components
                Y = X.dot(W[:, c:])
                X = Y.dot(W[:, c:].T)
                X = X + M
                X = X.reshape(2*ps + 1, 2*ps + 1, 2*ps + 1, n)

                # Overcomplete weighting
                if overcomplete:
                    w = 1.0 / (1.0 + n - c)
                    wei[i - ps: i + ps + 1,
                        j - ps: j + ps + 1,
                        k - ps: k + ps + 1] = wei[i - ps: i + ps + 1,
                                                  j - ps: j + ps + 1,
                                                  k - ps: k + ps + 1] + w
                    X = X * w
                    den[i - ps: i + ps + 1,
                        j - ps: j + ps + 1,
                        k - ps: k + ps + 1, :] = den[i - ps: i + ps + 1,
                                                     j - ps: j + ps + 1,
                                                     k - ps: k + ps + 1, :] + X
                    ncomps[i - ps: i + ps + 1,
                           j - ps: j + ps + 1,
                           k - ps: k + ps + 1] = ncomps[i - ps: i + ps + 1,
                                                        j - ps: j + ps + 1,
                                                        k - ps: k + ps + 1] + (n-c)*w
                    sig2[i - ps: i + ps + 1,
                           j - ps: j + ps + 1,
                           k - ps: k + ps + 1] = sig2[i - ps: i + ps + 1,
                                                      j - ps: j + ps + 1,
                                                      k - ps: k + ps + 1] + sig*w
                else:
                    den[i, j, k, :] = X[ps, ps, ps]
                    ncomps[i, j, k] = n - c
                    sig2[i, j, k] = sig

    if overcomplete:
        den = den / wei
        ncomps = ncoms / wei
        sig2 = sig2 / wei
    return den, np.sqrt(sig2), ncomps

def localpca(DWI, psize, nep):
    # performes localpca given the number of elements to be preserved
    m = (2*psize + 1) ** 3
    n = DWI.shape[3]
    DWIden = np.zeros(DWI.shape)
    for k in range(psize, DWI.shape[2] - psize):
        for j in range(psize, DWI.shape[1] - psize):
            for i in range(psize, DWI.shape[0] - psize):
                X = DWI[i - psize: i + psize + 1, j - psize: j + psize + 1,
                        k - psize: k + psize + 1, :]
                X = X.reshape(m, n)
                M = np.mean(X, axis=0)
                X = X - M
                [L, W] = np.linalg.eigh(np.dot(X.T, X)/m)
                Y = X.dot(W[:, -nep:])
                X = Y.dot(W[:, -nep:].T)
                X = X + M
                X = X.reshape(2*psize + 1, 2*psize + 1, 2*psize + 1, n)
                DWIden[i, j, k, :] = X[psize, psize, psize]
    return DWIden

def evaluate_directions(directions, gtdirections):
    angle = np.zeros(directions.shape[0:3])
    for i in range(directions.shape[0]):
        for j in range(directions.shape[1]):
            for k in range(directions.shape[0]):
                v1 = directions[i, j, k]
                vr = gtdirections[i, j, k]
                angle[i, j, k] = np.arccos(np.abs(v1.dot(vr.T))) * 180.0 / np.pi
    p1, p2, p3 = np.percentile(angle, [25 , 50, 75])
    return p2, p2 - p1, p3 - p2

def evaluate_metric(m, gtm):
    err = m - gtm
    p1, p2, p3 = np.percentile(err, [25, 50, 75])
    return p2, p2 - p1, p3 - p2

