# -*- coding: utf-8 -*-
"""
Created on June 02 2016

@author: Rafael Neto Henriques (rafaelnh21@gmail.com)
"""

# import relevant modules
import numpy as np


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


def pca_denoising(dwi, ps=2, overcomplete=True):
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
                        k - ps: k + ps + 1, :] = wei[i - ps: i + ps + 1,
                                                     j - ps: j + ps + 1,
                                                     k - ps: k + ps + 1, :] + w
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
        ncomps = ncomps / wei[..., 0]
        sig2 = sig2 / wei[..., 0]
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

