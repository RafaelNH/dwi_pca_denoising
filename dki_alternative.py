# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:42:14 2016

@author: Rafael
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.core.ndindex import ndindex

from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)

import scipy.optimize as opt

from dipy.reconst.utils import dki_design_matrix as design_matrix


def _dirfit_iter(inv_W, sig, min_signal=1.0e-6):
    """ Applies OLS fit of the direct MK estimation model.

    Parameters
    ----------
    inv_W : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: minimal signal in the data provided to `fit`.

    Returns
    -------
    All parameters estimated from the dir fit DKI model.
    Parameters are ordered as follows:
        1) Direct Mean Diffusivity measure
        2) Direct Mean Kurtosis measure
        3) Direct S0 estimate
    """
    log_s = np.log(np.maximum(sig, min_signal))
    params = np.dot(inv_W, log_s)

    return params


def ols_dki_df(gtab, data, mask=None, min_signal=1.0e-6):
    r""" Computes ordinary least squares (OLS) fit to calculate the
    mean kurtosis direct fit [1]_.

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.

    Returns
    -------
    params : ndarray ([X, Y, Z, ...], 3)
        All parameters estimated from the dir fit DKI model.
        Parameters are ordered as follows:
            1) Direct Mean Diffusivity measure
            2) Direct Mean Kurtosis measure
            3) Direct S0 estimate

    References
    ----------
       [1] Neto Henriques, R., Ferreira, H., Correia, M., 2012. Diffusion
           kurtosis imaging of the healthy human brain. Master Dissertation
           Bachelor and Master Program in Biomedical Engineering and
           Biophysics, Faculty of Science.
    """
    params = np.zeros(data.shape[:-1] + (3,))

    b = gtab.bvals
    W = np.zeros((len(b), 3))
    W[:, 0] = -b
    W[:, 1] = 1.0/6.0 * b**2
    W[:, 2] = np.ones(len(b))

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    inv_W = np.linalg.pinv(W)

    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            params[v] = _dirfit_iter(inv_W, data[v], min_signal=min_signal)
            params[v][1] = params[v][1] / (params[v][0]**2)
            params[v][2] = np.exp(params[v][2])
    return params


def _nls_df_err_func(params, design_matrix, data):
    """
    Error function for the non-linear least-squares fit of the tensor.

    Parameters
    ----------
    params : (3,)
        Parameters of the direct MD/MK estimate

    design_matrix : array
        The design matrix

    data : array
        The voxel signal in all gradient directions

    References
    ----------
    [1] Chang, L-C, Jones, DK and Pierpaoli, C (2005). RESTORE: robust
    estimation of tensors by outlier rejection. MRM, 53: 1088-95.
    """
    # This is the predicted signal given the params:
    y = np.exp(np.dot(design_matrix, params))

    # Compute the residuals
    return data - y


def _nls_df_iter(W, sig, params):
    """ Applies NLS fit of the direct MK estimation model.

    Parameters
    ----------
    W : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    params : float
        Initial estimate of parameters.

    Returns
    -------
    All parameters estimated from the dir fit DKI model.
    Parameters are ordered as follows:
        1) Direct Mean Diffusivity measure
        2) Direct Mean Kurtosis measure
        3) Direct S0 estimate
    """
    params, status = opt.leastsq(_nls_df_err_func, params, args=(W, sig))
    return params


def nls_dki_df(gtab, data, mask=None, min_signal=1.0e-6):
    r""" Computes non-linear least squares (NLS) fit to calculate the
    mean kurtosis direct fit [1]_.

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.

    Returns
    -------
    params : ndarray ([X, Y, Z, ...], 3)
        All parameters estimated from the dir fit DKI model.
        Parameters are ordered as follows:
            1) Direct Mean Diffusivity measure
            2) Direct Mean Kurtosis measure
            3) Direct S0 estimate

    References
    ----------
       [1] Neto Henriques, R., Ferreira, H., Correia, M., 2012. Diffusion
           kurtosis imaging of the healthy human brain. Master Dissertation
           Bachelor and Master Program in Biomedical Engineering and
           Biophysics, Faculty of Science.
    """
    params = np.zeros(data.shape[:-1] + (3,))

    b = gtab.bvals
    W = np.zeros((len(b), 3))
    W[:, 0] = -b
    W[:, 1] = 1.0/6.0 * b**2
    W[:, 2] = np.ones(len(b))

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    inv_W = np.linalg.pinv(W)

    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            params_ini = _dirfit_iter(inv_W, data[v], min_signal=min_signal)
            params[v] = _nls_df_iter(W, data[v], params_ini)
            params[v][1] = params[v][1] / (params[v][0]**2)
            params[v][2] = np.exp(params[v][2])
    return params


def avs_dki_df(gtab, data, mask=None, min_signal=1.0e-6):
    r""" Computes average signal of the direct fit [1]_.

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.

    Returns
    -------
    params : ndarray ([X, Y, Z, ...], 3)
        All parameters estimated from the dir fit DKI model.
        Parameters are ordered as follows:
            1) Direct Mean Diffusivity measure
            2) Direct Mean Kurtosis measure
            3) Direct S0 estimate
    """
    params = np.zeros(data.shape[:-1] + (3,))

    bmag = int(np.log10(gtab.bvals.max()))
    b = gtab.bvals.copy() / (10 ** (bmag-1))  # normalize b units
    b = b.round() * (10 ** (bmag-1))
    uniqueb = np.unique(b)
    nb = len(uniqueb)

    B = np.zeros((nb, 3))
    B[:, 0] = -uniqueb
    B[:, 1] = 1.0/6.0 * uniqueb**2
    B[:, 2] = np.ones(nb)

    ng = np.zeros(nb)
    for bi in range(nb):
        ng[bi] = np.sum(b == uniqueb[bi])
    ng = np.sqrt(ng)

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    index = ndindex(mask.shape)
    sig = np.zeros(nb)
    for v in index:
        if mask[v]:
            for bi in range(nb):
                sig[bi] = np.mean(data[v][b == uniqueb[bi]])
            # Define weights as diag(sqrt(ng) * yn**2)
            W = np.diag(ng * sig**2)
            BTW = np.dot(B.T, W)
            inv_BT_W_B = np.linalg.pinv(np.dot(BTW, B))
            invBTWB_BTW = np.dot(inv_BT_W_B, BTW)
            p = np.dot(invBTWB_BTW, np.log(sig))
            p[1] = p[1] / (p[0]**2)
            p[2] = np.exp(p[2])
            params[v] = p
    return params


def _wls_iter_gc(design_matrix, inv_design, sig, min_diffusivity, gc):
    """ Helper function used by wls_fit_dki - Applies WLS fit of the diffusion
    kurtosis model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients
    inv_design : array (g, 22)
        Inverse of the design matrix.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27, )
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """
    A = design_matrix

    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    ols_result = np.dot(inv_design, log_s)

    # Define weights as diag(yn**2)
    W = np.diag(gc**2 * np.exp(2 * np.dot(A, ols_result)))

    # DKI weighted linear least square solution
    inv_AT_W_A = np.linalg.pinv(np.dot(np.dot(A.T, W), A))
    AT_W_LS = np.dot(np.dot(A.T, W), log_s)
    wls_result = np.dot(inv_AT_W_A, AT_W_LS)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = wls_result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2
    KT_elements = wls_result[6:21] / MD_square

    # Write output
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements), axis=0)

    return dki_params


def wls_fit_dki(gtab, data, gc=None, mask=None, min_signal=1.0e-6):
    r""" Computes weighted linear least squares (WLS) fit to calculate
    the diffusion tensor and kurtosis tensor using a weighted linear
    regression diffusion kurtosis model [1]_.

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    gc : array (g,)
        Factores to correct different variances along different gradient
        directions. Default: np.ones(g) (for this case variance is assumed
        to be constant along different gradient directions)
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.

    Returns
    -------
    dki_params : array (N, 27)
        All parameters estimated from the diffusion kurtosis model for all N
        voxels.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """
    data = np.maximum(data, min_signal)

    params = np.zeros(data.shape[:-1] + (27,))

    B = design_matrix(gtab)
    inv_B = np.linalg.pinv(B)
    min_diffusivity = min_signal / -B.min()

    # Prepare gc
    if gc is None:
        gc = np.ones(gtab.bvals.size)

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            p = _wls_iter_gc(B, inv_B, data[v], min_diffusivity, gc)
            params[v] = p

    return params
