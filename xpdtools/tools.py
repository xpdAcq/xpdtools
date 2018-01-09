"""Tools for x-ray scattering data processing """
from itertools import starmap

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
from matplotlib.path import Path
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from scipy.integrate import simps
from skbeam.core.accumulators.binned_statistic import BinnedStatistic1D
from skbeam.core.mask import margin
from numba import jit

try:
    from diffpy.pdfgetx import PDFGetter
except ImportError:
    from xpdtools.shim import PDFGetterShim as PDFGetter


@jit(cache=True, nopython=True, nogil=True)
def mask_ring_median(values_array, positions_array, alpha):
    """Find outlier pixels in a single ring via a single pass with the median.

    Parameters
    ----------
    values_array : ndarray
        The ring values
    positions_array : ndarray
        The positions of the values
    alpha: float
        The threshold
    """
    z = np.abs(values_array - np.median(values_array)) / np.std(values_array)
    removals = positions_array[np.where(z > alpha)]
    return removals


def mask_ring_mean(values_array, positions_array, alpha):
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
    """
    m = np.ones_like(positions_array, dtype=bool)
    removals = []
    while len(values_array) > 0:
        m[np.in1d(positions_array, removals)] = False
        v = values_array[m]
        std = np.std(v)
        norm_v_list = np.abs(v - np.mean(v)) / std
        if np.all(norm_v_list < alpha) or std == 0.0:
            break
        # get the index of the worst pixel
        worst_idx = np.argmax(norm_v_list)
        # add the worst position to the mask
        removals.append(positions_array[m][worst_idx])
    return removals


mask_ring_dict = {'median': mask_ring_median, 'mean': mask_ring_mean}


def binned_outlier(img, binner, alpha=3, tmsk=None, mask_method='median'):
    """Sigma Clipping based masking

    Parameters
    ----------
    img : np.ndarray
        The image
    binner : BinnedStatistic1D instance
        The binned statistics information
    alpha : float
        The number of standard deviations to clip
    tmsk : np.ndarray, optional
        Prior mask. If None don't use a prior mask, defaults to None.
    mask_method : {'median', 'mean'}, optional
        The method to use for creating the mask, median is faster, mean is more
        accurate. Defaults to median.

    Returns
    -------
    np.ndarray:
        The mask
    """
    print('start mask')

    idx = binner.argsort_index
    vfs = img.flatten()[idx]
    pfs = np.arange(np.size(img))[idx]
    h = binner.flatcount
    t = []
    i = 0
    for j, k in enumerate(h):
        if k > 0:
            t.append((vfs[i: i + k], pfs[i: i + k], alpha))
        i += k
    p_err = np.seterr(all='ignore')
    removals = starmap(mask_ring_dict[mask_method], t)
    np.seterr(**p_err)
    removals = [item for sublist in removals for item in sublist]
    if tmsk is None:
        tmsk = np.ones(img.shape, dtype=bool)
    tmsk = tmsk.ravel()
    tmsk[removals] = False
    tmsk = tmsk.reshape(img.shape)
    print('finished mask')
    return tmsk.astype(bool)


def mask_img(img, binner,
             edge=30,
             lower_thresh=0.0,
             upper_thresh=None,
             bs_width=13, tri_offset=13, v_asym=0,
             alpha=3,
             auto_type='median',
             tmsk=None):
    """
    Mask an image based off of various methods

    Parameters
    ----------
    img: np.ndarray
        The image to be masked
    binner: pyFAI.geometry.Geometry
        The pyFAI description of the detector orientation or any
        subclass of pyFAI.geometry.Geometry class
    edge: int, optional
        The number of edge pixels to mask. Defaults to 30. If None, no edge
        mask is applied
    lower_thresh: float, optional
        Pixels with values less than or equal to this threshold will be masked.
        Defaults to 0.0. If None, no lower threshold mask is applied
    upper_thresh: float, optional
        Pixels with values greater than or equal to this threshold will be
        masked.
        Defaults to None. If None, no upper threshold mask is applied.
    bs_width: int, optional
        The width of the beamstop in pixels. Defaults to 13.
        If None, no beamstop polygon mask is applied.
    tri_offset: int, optional
        The triangular pixel offset to create a pointed beamstop polygon mask.
        Defaults to 13. If None, no beamstop polygon mask is applied.
    v_asym: int, optional
        The vertical asymmetry of the polygon beamstop mask. Defaults to 0.
        If None, no beamstop polygon mask is applied.
    alpha: float or tuple or, 1darray, optional
        Then number of acceptable standard deviations, if tuple then we use
        a linear distribution of alphas from alpha[0] to alpha[1], if array
        then we just use that as the distribution of alphas. Defaults to 2.5.
        If None, no outlier masking applied.
    tmsk: np.ndarray, optional
        The starting mask to be compounded on. Defaults to None. If None mask
        generated from scratch.

    Returns
    -------
    tmsk: np.ndarray
        The mask as a boolean array. True pixels are good pixels, False pixels
        are masked out.

    """

    if tmsk is None:
        working_mask = np.ones(img.shape).astype(bool)
    else:
        working_mask = tmsk.copy()
    if edge:
        working_mask *= margin(img.shape, edge)
    if lower_thresh:
        working_mask *= (img >= lower_thresh).astype(bool)
    if upper_thresh:
        working_mask *= (img <= upper_thresh).astype(bool)
    if all([a is not None for a in [bs_width, tri_offset, v_asym]]):
        center_x, center_y = [binner.getFit2D()[k] for k in
                              ['centerX', 'centerY']]
        nx, ny = img.shape
        mask_verts = [(center_x - bs_width, center_y),
                      (center_x, center_y - tri_offset),
                      (center_x + bs_width, center_y),
                      (center_x + bs_width + v_asym, ny),
                      (center_x - bs_width - v_asym, ny)]

        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(mask_verts)
        grid = path.contains_points(points)
        # Plug msk_grid into into next (edge-mask) step in automask
        working_mask *= ~grid.reshape((ny, nx))

    if alpha:
        working_mask *= binned_outlier(img, binner, alpha=alpha,
                                       tmsk=working_mask,
                                       mask_method=auto_type)
    working_mask = working_mask.astype(np.bool)
    return working_mask


def generate_binner(geo, img_shape=None, mask=None):
    """Create a pixel resolution BinnedStats1D instance

    Parameters
    ----------
    geo : pyFAI.geometry.Geometry instance
        The calibrated geometry
    img_shape : tuple, optional
        The shape of the image, if None pull from the mask. Defaults to None.
    mask : ndarray, optional
        The mask to be applied, if None no mask is applied. Defaults to None.

    Returns
    -------
    BinnedStatistic1D :
        The configured instance of the binner.
    """
    if img_shape is None:
        img_shape = mask.shape
    r = geo.rArray(img_shape)
    q = geo.qArray(img_shape) / 10  # type: np.ndarray
    q_dq = geo.deltaQ(img_shape) / 10  # type: np.ndarray

    pixel_size = [getattr(geo, a) for a in ['pixel1', 'pixel2']]
    rres = np.hypot(*pixel_size)
    rbins = np.arange(np.min(r) - rres / 2., np.max(r) + rres / 2., rres / 2.)
    # This is only called once, use the function version`
    rbinned = BinnedStatistic1D(r.ravel(), statistic=np.max, bins=rbins, )

    qbin_sizes = rbinned(q_dq.ravel())
    qbin_sizes = np.nan_to_num(qbin_sizes)
    qbin = np.cumsum(qbin_sizes)
    qbin[0] = np.min(q_dq)
    if mask is not None:
        mask = mask.flatten()
    return BinnedStatistic1D(q.flatten(), bins=qbin, mask=mask)


def z_score_image(img, binner):
    """Z score an image according to the azimuthal average

    Parameters
    ----------
    img : ndarray
        The image
    binner : BinnedStatistic1D instance
        The binner
    Returns
    -------
    ndarray :
        The z scored image
    """
    idx = binner.argsort_index

    vfs = img.flatten()[idx]

    i = 0
    for j, k in enumerate(binner.flatcount):
        if k > 0:
            vfs[i: i + k] -= np.mean(vfs[i: i + k])
            vfs[i: i + k] /= np.std(vfs[i: i + k])
        i += k

    img2 = np.empty(vfs.shape)
    img2[idx] = vfs
    img2 = np.nan_to_num(img2)

    return img2.reshape(img.shape)


def polarization_correction(img, geo, polarization_factor=.99):
    """Perform polarization correction on an image

    Parameters
    ----------
    img : ndarray
        The image
    geo : pyFAI.geometry.Geometry instance
        The calibrated geometry
    polarization_factor : float
        The polarization factor to apply

    Returns
    -------
    ndarray :
        The corrected image
    """
    return img / geo.polarization(img.shape, polarization_factor)


def load_geo(cal_params):
    """Load a pyFAI geometry from a dict of calibration parameters

    Parameters
    ----------
    cal_params : dict
        The calibration parameters

    Returns
    -------
    AzimuthalIntegrator :
        The calibrate azimuthal integrator (which inherits from the geometry)
    """
    ai = AzimuthalIntegrator()
    ai.setPyFAI(**cal_params)
    return ai


def pdf_getter(x, y, composition, **kwargs):
    """Process the data to the PDF

    Parameters
    ----------
    x : ndarray
        The q or tth values
    y : ndarray
        The scattered intensity
    composition : str
        The composition
    kwargs: dict
        Additional kwargs for PDFGetter

    Returns
    -------
    r : ndarray
        The radial values
    gr: ndarray
        The PDF
    config: dict
        The PDFGetter config
    """
    pg = PDFGetter()
    kwargs.update({'composition': composition})
    args = (x, y)
    res = pg(*args, **kwargs)
    return res[0], res[1], pg.config


def fq_getter(x, y, composition, **kwargs):
    """Process the data to F(Q)

    Parameters
    ----------
    x : ndarray
        The q or tth values
    y : ndarray
        The scattered intensity
    composition : str
        The composition
    kwargs: dict
        Additional kwargs for PDFGetter

    Returns
    -------
    q : ndarray
        The radial values
    fq: ndarray
        The reduced structure function
    config: dict
        The PDFGetter config
    """
    pg = PDFGetter()
    kwargs.update({'composition': composition})
    args = (x, y)
    pg(*args, **kwargs)
    res = pg.fq
    return res[0], res[1], pg.config


def overlay_mask(img, mask):
    """Overlay mask on image, masked pixels are ``np.nan``"""
    img2 = img.copy()
    img2[~mask] = np.nan
    return img2


def nu_fq_getter(q, iq, composition, **kwargs):
    """Process the data to F(Q) on a non uniform grid

    Parameters
    ----------
    x : ndarray
        The q or tth values
    y : ndarray
        The scattered intensity
    composition : str
        The composition
    kwargs: dict
        Additional kwargs for PDFGetter

    Returns
    -------
    q : ndarray
        The radial values
    fq: ndarray
        The reduced structure function
    config: dict
        The PDFGetter config
    """
    kwargs.update({'composition': composition})
    # explicit qmin/qmaxinst cutting
    truth_values = np.where((kwargs['qmaxinst'] > q) & (q > kwargs['qmin']))
    pg = PDFGetter()
    # remove resampling transformations (and bg sub)
    for t in [7, 6, 1]:
        pg.transformations.pop(t)
    pg(q[truth_values], iq[truth_values], **kwargs)
    res = pg.fq
    return res[0], res[1], pg.config


def nu_pdf_getter(q, fq):
    """Process a non uniform F(Q) to the PDF

    Parameters
    ----------
    q : ndarray
        The q or tth values
    fq : ndarray
        The reduced structure funciton

    Returns
    -------
    r : ndarray
        The radial values
    gr: ndarray
        The PDF
    """
    rgrid = np.arange(0, 30.01, np.pi / np.max(q))
    dgr = 2 / np.pi * fq * np.sin(q * rgrid[:, np.newaxis])
    gr = simps(dgr, q)
    return rgrid, gr
