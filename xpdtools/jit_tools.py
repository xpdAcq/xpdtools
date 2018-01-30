"""Just in time compiled tools (seperated from tools so we don't keep
compiling them"""
##############################################################################
#
# xpdtools            by Billinge Group
#                   Simon J. L. Billinge sb2896@columbia.edu
#                   (c) 2017 trustees of Columbia University in the City of
#                        New York.
#                   All rights reserved
#
# File coded by:    Christopher J. Wright
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
import numpy as np
from numba import jit, boolean


@jit(cache=True, nopython=True, nogil=True)
def mask_ring_median(values_array, positions_array, alpha):  # pragma: no cover
    """Find outlier pixels in a single ring via a single pass with the median.

    Parameters
    ----------
    values_array : ndarray
        The ring values
    positions_array : ndarray
        The positions of the values
    alpha: float
        The threshold

    Returns
    -------
    removals: np.ndarray
        The positions of pixels to be removed from the data
    """
    z = np.abs(values_array - np.median(values_array)) / np.std(values_array)
    removals = positions_array[np.where(z > alpha)]
    return removals


@jit(cache=True, nopython=True, nogil=True)
def mask_ring_mean(values_array, positions_array, alpha):  # pragma: no cover
    """Find outlier pixels in a single ring via a pixel by pixel method with
    the mean.

    Parameters
    ----------
    values_array : ndarray
        The ring values
    positions_array : ndarray
        The positions of the values
    alpha: float
        The threshold

    Returns
    -------
    removals: np.ndarray
        The positions of pixels to be removed from the data
    """
    m = np.ones(positions_array.shape, dtype=boolean)
    removals = []
    while True:
        b = np.array([item in removals for item in positions_array])
        m[b] = False
        v = values_array[m]
        if len(v) <= 1:
            break
        std = np.std(v)
        if std == 0.0:
            break
        norm_v_list = np.abs(v - np.mean(v)) / std
        if np.all(norm_v_list < alpha):
            break
        # get the index of the worst pixel
        worst_idx = np.argmax(norm_v_list)
        # add the worst position to the mask
        removals.append(positions_array[m][worst_idx])
    return removals
