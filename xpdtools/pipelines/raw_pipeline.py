"""Main pipeline chunks for processing images to I(Q) and PDF"""
import operator as op

import numpy as np
from skbeam.core.utils import q_to_twotheta
from rapidz import Stream

from xpdtools.calib import img_calibration
from xpdtools.tools import (
    load_geo,
    mask_img,
    map_to_binner,
    fq_getter,
    pdf_getter,
    sq_getter,
    generate_map_bin,
    pluck_check,
    splay_tuple,
    call_stream_element,
    check_kwargs,
)

namespace = dict(
    raw_foreground=Stream(stream_name="raw foreground"),
    raw_foreground_dark=Stream(stream_name="raw foreground dark"),
    raw_background=Stream(stream_name="raw background"),
    raw_background_dark=Stream(stream_name="raw background dark"),
    wavelength=Stream(stream_name="wavelength"),
    calibrant=Stream(stream_name="calibrant"),
    detector=Stream(stream_name="detector"),
    is_calibration_img=Stream(stream_name="Is Calibration"),
    geo_input=Stream(stream_name="geometry"),
    img_counter=Stream(stream_name="img counter"),
    composition=Stream(stream_name="composition"),
    polarization_factor=.99,
    mask_setting={"setting": "auto"},
    calib_setting={"setting": True},
    bg_scale=1,
)


def image_process(
    raw_foreground,
    raw_foreground_dark,
    raw_background,
    raw_background_dark,
    bg_scale=1.,
    **kwargs
):
    """Pipeline chunk to perform image processing, including dark and
    background subtraction.

    Parameters
    ----------
    raw_foreground : Stream
    raw_foreground_dark : Stream
    raw_background : Stream
    raw_background_dark : Stream
    bg_scale : float, optional
        The background scale factor. Defaults to 1

    Returns
    -------
    ns : dict
        The namespace created by the chunk
    """
    # Get the image shape for the binner
    dark_corrected_foreground = raw_foreground.combine_latest(
        raw_foreground_dark, emit_on=0
    ).starmap(op.sub)
    dark_corrected_background = (
        raw_background.combine_latest(raw_background_dark, emit_on=0)
        .starmap(op.sub)
        .map(op.mul, bg_scale)
    )
    bg_corrected_img = dark_corrected_foreground.combine_latest(
        dark_corrected_background, emit_on=0
    ).starmap(op.sub, stream_name="background corrected img")
    img_shape = bg_corrected_img.map(np.shape).unique(history=1)
    return locals()


def calibration(
    wavelength,
    calibrant,
    detector,
    is_calibration_img,
    geo_input,
    bg_corrected_img,
    img_shape,
    calib_setting=None,
    **kwargs
):
    """Pipeline chunk for performing and loading calibration

    Parameters
    ----------
    wavelength : Stream
    calibrant : Stream
    detector : Stream
    is_calibration_img : Stream
    geo_input : Stream
    bg_corrected_img : Stream
    img_shape : Stream
    calib_setting : None or dict, optional
        The calibration setting, if set to ``{"setting": False}`` the user
        will not be prompted to perform calibration on calibration samples.
        This is useful for not performing calibration when re analyzing an
        entire experiment.

    Returns
    -------
    ns : dict
        The namespace created by the chunk

    """
    # Calibration management

    if calib_setting is None:
        calib_setting = {"setting": True}
    gated_cal = (
        bg_corrected_img.combine_latest(is_calibration_img, emit_on=0)
        .filter(pluck_check, 1)
        .pluck(0, stream_name="Gate calibration")
    )

    gen_geo_cal = (
        gated_cal.combine_latest(wavelength, calibrant, detector, emit_on=0)
        .filter(check_kwargs, "setting", True, **calib_setting)
        .starmap(img_calibration)
    )

    gen_geo = gen_geo_cal.pluck(1)

    geometry = (
        geo_input.combine_latest(is_calibration_img, emit_on=0)
        .filter(pluck_check, 1, False)
        .pluck(0, stream_name="Gate calibration")
        .map(load_geo)
        .union(gen_geo, stream_name="Combine gen and load cal")
    )

    # Image corrections
    geometry_img_shape = geometry.zip_latest(img_shape)

    # Only create map and bins (which is expensive) when needed
    # (new calibration)
    map_res = geometry_img_shape.starmap(generate_map_bin)
    cal_binner = map_res.starmap(map_to_binner)
    return locals()


def scattering_correction(
    geometry, img_shape, bg_corrected_img, polarization_factor=.99, **kwargs
):
    """Pipeline chunk for performing scattering corrections on images,
    including the polarization correction.

    Parameters
    ----------
    geometry : Stream
    img_shape : Stream
    bg_corrected_img : Stream
    polarization_factor : float, optional
        The polarization factor used to correct the image. Defaults to .99

    Returns
    -------
    ns : dict
        The namespace created by the chunk
    """

    polarization_callable = geometry.map(getattr, "polarization")

    polarization_array = polarization_callable.zip_latest(img_shape).starmap(
        call_stream_element, polarization_factor
    )

    pol_correction_combine = bg_corrected_img.combine_latest(
        polarization_array, emit_on=bg_corrected_img
    )
    pol_corrected_img = pol_correction_combine.starmap(op.truediv)
    return locals()


def gen_mask(
    pol_corrected_img,
    cal_binner,
    img_counter,
    mask_setting=None,
    mask_kwargs=None,
    **kwargs
):
    """Pipeline chunk for creating masks

    Parameters
    ----------
    pol_corrected_img : Stream
    cal_binner : Stream
    img_counter : Stream
    mask_setting : dict, optional
        The setting for the frequency of the mask. If set to
        ``{'setting': 'auto'}`` each image gets a mask generated for it, if
        set to ``{'setting': 'first'}`` only the first image in the series has
        a mask generated for it and all subsequent images in the series use
        that mask, if set to ``{'setting': 'none'}`` then no image is masked.
        Defaults to ``{'setting': 'auto'}``.
    mask_kwargs : dict, optional
        The keyword arguments passed to ``xpdtools.tools.mask_img``.
        Defaults to ``dict(edge=30, lower_thresh=0.0, upper_thresh=None,
         alpha=3, auto_type="median", tmsk=None,)``

    Returns
    -------
    ns : dict
        The namespace created by the chunk

    """
    if mask_kwargs is None:
        mask_kwargs = dict(
            edge=30,
            lower_thresh=0.0,
            upper_thresh=None,
            alpha=3,
            auto_type="median",
            tmsk=None,
        )
    if mask_setting is None:
        mask_setting = {"setting": "auto"}
    # emit on img so we don't propagate old image data
    # note that the pol_corrected_img has touched the geometry and so always
    # comes after the geometry itself, so we never have a condition where
    # we fail to emit because pol_corrected_img comes down first
    img_cal_binner = pol_corrected_img.combine_latest(
        cal_binner, emit_on=pol_corrected_img
    )

    all_mask_filter = img_cal_binner.filter(
        check_kwargs, "setting", "auto", **mask_setting
    )
    all_mask = all_mask_filter.starmap(
        mask_img, stream_name="mask", **mask_kwargs
    )
    first_mask_filter = img_cal_binner.filter(
        check_kwargs, "setting", "first", **mask_setting
    )
    first_mask = (
        first_mask_filter.zip(img_counter)
        .filter(pluck_check, position=1, eq=1)
        .pluck(0)
        .starmap(
            mask_img,
            stream_name="mask",
            **dict(
                edge=30,
                lower_thresh=0.0,
                upper_thresh=None,
                alpha=3,
                auto_type="median",
                tmsk=None,
            )
        )
    )

    no_mask_filter = img_cal_binner.filter(
        check_kwargs, "setting", "none", **mask_setting
    )
    no_mask = no_mask_filter.pluck(0).map(np.shape).map(np.ones, dtype=bool)

    mask = all_mask.union(first_mask, no_mask)

    mask_kwargs = all_mask.kwargs
    first_mask.kwargs = mask_kwargs

    mask_setting = all_mask_filter.kwargs
    first_mask_filter.kwargs = mask_setting
    no_mask_filter.kwargs = mask_setting

    return locals()


def integration(map_res, mask, wavelength, pol_corrected_img, **kwargs):
    """Pipeline chunk for computing azimuthal integration

    Parameters
    ----------
    map_res : Stream
    mask : Stream
    wavelength : Stream
    pol_corrected_img : Stream

    Returns
    -------
    ns : dict
        The namespace created by the chunk
    """
    # Integration
    binner = (
        map_res.combine_latest(mask, emit_on=1)
        .map(splay_tuple)
        .starmap(map_to_binner)
    )
    q = binner.map(getattr, "bin_centers", stream_name="Q")
    tth = (
        q.combine_latest(wavelength, emit_on=0)
        .starmap(q_to_twotheta, stream_name="tth")
        .map(np.rad2deg)
    )

    f_img_binner = binner.combine_latest(
        pol_corrected_img.map(np.ravel), emit_on=1
    )

    mean = f_img_binner.starmap(
        call_stream_element, statistic="mean", stream_name="Mean IQ"
    ).map(np.nan_to_num)
    return locals()


def pdf_gen(q, mean, composition, **kwargs):
    """Pipeline chunk for computing the Structure Factor S(Q), Reduced
    Structure Factor F(Q) and Atomic Pair Distribution Function (PDF).

    Parameters
    ----------
    q : Stream
    mean : Stream
    composition : Stream
    kwargs : Any
         The keyword arguments passed to the PDF creation, please see PDFgetx3
         for more details

    Returns
    -------
    ns : dict
        The namespace created by the chunk
    """

    # PDF
    iq_comp = q.combine_latest(mean, emit_on=1).combine_latest(
        composition, emit_on=0
    )
    iq_comp_map = iq_comp.map(splay_tuple)

    # TODO: split these all up into their components ((r, pdf), (q, fq)...)
    sq = iq_comp_map.starmap(
        sq_getter,
        stream_name="sq",
        **(dict(dataformat="QA", qmaxinst=28, qmax=25))
    )
    fq = iq_comp_map.starmap(
        fq_getter,
        stream_name="fq",
        **(dict(dataformat="QA", qmaxinst=28, qmax=25))
    )
    pdf = iq_comp_map.starmap(
        pdf_getter,
        stream_name="pdf",
        **(dict(dataformat="QA", qmaxinst=28, qmax=22))
    )
    fq_kwargs = fq.kwargs
    sq.kwargs = fq_kwargs
    pdf_kwargs = pdf.kwargs

    return locals()


pipeline_order = [
    image_process,
    calibration,
    scattering_correction,
    gen_mask,
    integration,
    pdf_gen,
]
