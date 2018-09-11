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

mask_setting = {"setting": "auto"}
calib_setting = {"setting": True}
raw_foreground = Stream(stream_name="raw foreground")
raw_foreground_dark = Stream(stream_name="raw foreground dark")
raw_background = Stream(stream_name="raw background")
raw_background_dark = Stream(stream_name="raw background dark")

# Get the image shape for the binner
dark_corrected_foreground = raw_foreground.combine_latest(
    raw_foreground_dark, emit_on=0
).starmap(op.sub)
dark_corrected_background = raw_background.combine_latest(
    raw_background_dark, emit_on=0
).starmap(op.sub)
bg_corrected_img = dark_corrected_foreground.combine_latest(
    dark_corrected_background, emit_on=0
).starmap(op.sub, stream_name="background corrected img")

# Calibration management
wavelength = Stream(stream_name="wavelength")
calibrant = Stream(stream_name="calibrant")
detector = Stream(stream_name="detector")
is_calibration_img = Stream(stream_name="Is Calibration")
geo_input = Stream(stream_name="geometry")
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
    .filter(pluck_check, 1)
    .pluck(0, stream_name="Gate calibration")
    .map(load_geo)
    .union(gen_geo, stream_name="Combine gen and load cal")
)

# Image corrections
img_shape = bg_corrected_img.map(np.shape).unique(history=1)
geometry_img_shape = geometry.zip_latest(img_shape)

# Only create map and bins (which is expensive) when needed (new calibration)
map_res = geometry_img_shape.starmap(generate_map_bin)
cal_binner = map_res.starmap(map_to_binner)

polarization_callable = geometry.map(getattr, "polarization")

polarization_array = polarization_callable.zip_latest(img_shape).starmap(
    call_stream_element, .99
)

pol_correction_combine = bg_corrected_img.combine_latest(
    polarization_array, emit_on=bg_corrected_img
)
pol_corrected_img = pol_correction_combine.starmap(op.truediv)

# emit on img so we don't propagate old image data
# note that the pol_corrected_img has touched the geometry and so always comes
# after the geometry itself, so we never have a condition where  we fail to
# emit because pol_corrected_img comes down first
img_cal_binner = pol_corrected_img.combine_latest(
    cal_binner, emit_on=pol_corrected_img
)

all_mask_filter = img_cal_binner.filter(
    check_kwargs, "setting", "auto", **mask_setting
)
all_mask = all_mask_filter.starmap(
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
img_counter = Stream(stream_name="img counter")
first_mask_filter = img_cal_binner.filter(
    check_kwargs, "setting", "first", **mask_setting
)
first_mask = (
    first_mask_filter.zip(img_counter)
    .filter(pluck_check, eq=1)
    .pluck(0)
    .starmap(mask_img, stream_name="mask", **{})
)

no_mask_filter = img_cal_binner.filter(
    check_kwargs, "setting", "none", **mask_setting
)
no_mask = no_mask_filter.pluck(0).starmap(np.ones, dtype=bool)

mask = all_mask.union(first_mask, no_mask)

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

# PDF
composition = Stream(stream_name="composition")
iq_comp = q.combine_latest(mean, emit_on=1).combine_latest(
    composition, emit_on=0
)
iq_comp_map = iq_comp.map(splay_tuple)

# TODO: split these all up into their components ((r, pdf), (q, fq)...)
sq = iq_comp_map.starmap(
    sq_getter,
    stream_name="sq",
    **(dict(dataformat="QA", qmaxinst=28, qmax=25, rstep=np.pi / 25))
)
fq = iq_comp_map.starmap(
    fq_getter,
    stream_name="fq",
    **(dict(dataformat="QA", qmaxinst=28, qmax=25, rstep=np.pi / 25))
)
pdf = iq_comp_map.starmap(
    pdf_getter,
    stream_name="pdf",
    **(dict(dataformat="QA", qmaxinst=28, qmax=22, rstep=np.pi / 22))
)

# All the kwargs

# Tie all the kwargs together (so changes in one node change the rest)
mask_kwargs = all_mask.kwargs
first_mask.kwargs = mask_kwargs

mask_setting = all_mask_filter.kwargs


fq_kwargs = fq.kwargs
sq.kwargs = fq_kwargs
pdf_kwargs = pdf.kwargs
