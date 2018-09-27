"""Main pipeline for processing images to I(Q) and PDF"""
import operator as op

import numpy as np
from skbeam.core.utils import q_to_twotheta
from streamz_ext import Stream

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
    # Calibration management
    geometry = (geo_input.map(load_geo))

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
    mask_kwargs=None,
    **kwargs
):
    if mask_kwargs is None:
        mask_kwargs = dict(
            edge=30,
            lower_thresh=0.0,
            upper_thresh=None,
            alpha=3,
            auto_type="median",
            tmsk=None,
        )
    # emit on img so we don't propagate old image data
    # note that the pol_corrected_img has touched the geometry and so always
    # comes after the geometry itself, so we never have a condition where
    # we fail to emit because pol_corrected_img comes down first
    img_cal_binner = pol_corrected_img.combine_latest(
        cal_binner, emit_on=pol_corrected_img
    )

    mask = img_cal_binner.starmap(
        mask_img, stream_name="mask", **mask_kwargs
    )
    mask_kwargs = mask.kwargs
    return locals()


def integration(map_res, mask, wavelength, pol_corrected_img, **kwargs):
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

    p = pol_corrected_img.map(np.ravel)
    f_img_binner = binner.combine_latest(
        p, emit_on=1
    )

    mean = f_img_binner.starmap(
        call_stream_element, statistic="mean", stream_name="Mean IQ"
    ).map(np.nan_to_num)
    return locals()


def pdf_gen(q, mean, composition, **kwargs):

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
