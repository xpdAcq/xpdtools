"""Main pipeline for processing images to I(Q) and PDF"""
import operator as op

import numpy as np
from skbeam.core.utils import q_to_twotheta
from streamz_ext import Stream

from xpdtools.calib import img_calibration
from xpdtools.tools import (z_score_image, load_geo, mask_img, generate_binner,
                            overlay_mask,
                            fq_getter, pdf_getter, sq_getter)

mask_setting = {'setting': 'auto'}
raw_foreground = Stream(stream_name='raw foreground')
raw_foreground_dark = Stream(stream_name='raw foreground dark')
raw_background = Stream(stream_name='raw background')
raw_background_dark = Stream(stream_name='raw background dark')
img_shape = (raw_foreground.
             union(raw_foreground_dark, raw_background, raw_background_dark).
             map(np.shape).
             unique(history=1))
# Get the image shape for the binner
dark_corrected_foreground = (
    raw_foreground.
    combine_latest(raw_foreground_dark, emit_on=0).
    starmap(op.sub)
)
dark_corrected_background = (
    raw_background.
    combine_latest(raw_background_dark, emit_on=0).
    starmap(op.sub)
)
bg_corrected_img = (
    dark_corrected_foreground.
    combine_latest(dark_corrected_background, emit_on=0).
    starmap(op.sub, stream_name='background corrected img')
)

# Calibration management
wavelength = Stream(stream_name='wavelength')
calibrant = Stream(stream_name='calibrant')
detector = Stream(stream_name='detector')
is_calibration_img = Stream(stream_name='Is Calibration')
geo_input = Stream(stream_name='geometry')
gated_cal = (
    bg_corrected_img.
    combine_latest(is_calibration_img, emit_on=0).
    filter(lambda a: bool(a[1])).
    pluck(0, stream_name='Gate calibration'))

gen_geo_cal = (
    gated_cal.
    combine_latest(wavelength,
                   calibrant,
                   detector, emit_on=0).
    map(img_calibration)
)

gen_geo = gen_geo_cal.pluck(1)

geometry = (
    geo_input.combine_latest(is_calibration_img, emit_on=0).
    filter(lambda a: not bool(a[1])).
    pluck(0, stream_name='Gate calibration').
    map(load_geo).
    union(gen_geo, stream_name='Combine gen and load cal'))

# Image corrections
geometry_img_shape = geometry.zip_latest(img_shape)

polarization_array = (
    geometry_img_shape.
    starmap(lambda geo, shape, polarization_factor: geo.polarization(
        shape, polarization_factor), .99))

pol_correction_combine = (
    bg_corrected_img
    .combine_latest(polarization_array, emit_on=bg_corrected_img))
pol_corrected_img = pol_correction_combine.starmap(op.truediv)


# Only create binner (which is expensive) when needed (new calibration)
cal_binner = (geometry_img_shape
              .starmap(generate_binner))

img_cal_binner = (
    pol_corrected_img.
    combine_latest(cal_binner))

all_mask = (
    img_cal_binner
    .filter(lambda x, **kwargs: mask_setting['setting'] == 'auto')
    .starmap(mask_img, stream_name='mask', **{})
)
img_counter = Stream()
first_mask = (
    img_cal_binner
    .filter(lambda x, **kwargs: mask_setting['setting'] == 'first')
    .zip(img_counter)
    .filter(lambda x: x[1] == 1).pluck(0)
    .starmap(mask_img, stream_name='mask', **{})
)

no_mask = (
    img_cal_binner
    .filter(lambda x, **kwargs: mask_setting['setting'] == 'none')
    .starmap(lambda img, *_: np.ones(img.shape, dtype=bool))
)

mask = all_mask.union(first_mask, no_mask)

# Integration
# TODO: try to get this to not call pyFAI again
binner = (
    mask.
    combine_latest(geometry, emit_on=0).
    starmap(lambda mask, geo: generate_binner(geo, mask=mask)))
f_img_binner = pol_corrected_img.map(np.ravel).combine_latest(binner,
                                                              emit_on=0)

mean = (
    f_img_binner.
    starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
            statistic='mean').map(np.nan_to_num))
median = (
    f_img_binner.
    starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
            statistic='median').map(np.nan_to_num))
std = (
    f_img_binner.
    starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
            statistic='std').map(np.nan_to_num))

q = binner.map(getattr, 'bin_centers')
tth = (
    q.combine_latest(wavelength, emit_on=0)
    .starmap(q_to_twotheta, stream_name='tth'))

z_score = (
    pol_corrected_img.
    combine_latest(binner, emit_on=0).
    starmap(z_score_image, stream_name='z score').
    combine_latest(mask, emit_on=0).starmap(overlay_mask))

# PDF
composition = Stream(stream_name='composition')
iq_comp = (
    q.zip(mean)
    .combine_latest(composition, emit_on=0))
iq_comp_map = (iq_comp.map(lambda x: (x[0][0], x[0][1], x[1])))
sq = iq_comp_map.starmap(sq_getter, stream_name='sq', **(
    dict(dataformat='QA', qmaxinst=28, qmax=25, rstep=np.pi / 25)))
fq = iq_comp_map.starmap(fq_getter, stream_name='fq', **(
    dict(dataformat='QA', qmaxinst=28, qmax=25, rstep=np.pi / 25)))
pdf = iq_comp_map.starmap(pdf_getter, stream_name='pdf', **(
    dict(dataformat='QA', qmaxinst=28, qmax=22, rstep=np.pi / 22)))

# All the kwargs

# Tie all the kwargs together (so changes in one node change the rest)
mask_kwargs = all_mask.kwargs
first_mask.kwargs = mask_kwargs
no_mask.kwargs = mask_kwargs

fq_kwargs = fq.kwargs
sq.kwargs = fq_kwargs
pdf_kwargs = pdf.kwargs
