"""Main pipeline for processing images to I(Q) and PDF"""
import operator as op

import numpy as np
from skbeam.core.utils import q_to_twotheta
from streamz_ext import Stream

from xpdtools.calib import img_calibration
from xpdtools.tools import (z_score_image, load_geo, mask_img, generate_binner,
                            overlay_mask,
                            fq_getter, pdf_getter)

mask_setting = {'setting': 'auto'}
# Default kwargs
mask_kwargs = {}
fq_kwargs = dict(dataformat='QA', qmaxinst=28, qmax=25, rstep=np.pi / 25)
pdf_kwargs = dict(dataformat='QA', qmaxinst=28, qmax=22, rstep=np.pi / 22)

# Detector corrections
raw_foreground = Stream(stream_name='raw foreground')
raw_foreground_dark = Stream(stream_name='raw foreground dark')
raw_background = Stream(stream_name='raw background')
raw_background_dark = Stream(stream_name='raw background dark')

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
img_shape = bg_corrected_img.map(np.shape).unique(history=1)
geometry_img_shape = geometry.zip_latest(img_shape)

polarization_array = (
    geometry_img_shape.
    starmap(lambda geo, shape, polarization_factor: geo.polarization(
        shape, polarization_factor), .99))

pol_corrected_img_zip = (
    bg_corrected_img.
    combine_latest(geometry, emit_on=0))
pol_correction_combine = (bg_corrected_img
    .combine_latest(polarization_array, emit_on=0))
pol_corrected_img = pol_correction_combine.starmap(op.truediv)


# Only create binner (which is expensive) when needed (new calibration)
cal_binner = (geometry_img_shape
              .starmap(generate_binner))

mask = (
    pol_corrected_img.
    combine_latest(cal_binner).
    starmap(mask_img, stream_name='mask', **mask_kwargs))

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
            statistic='mean'))
median = (
    f_img_binner.
    starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
            statistic='median'))
std = (
    f_img_binner.
    starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
            statistic='std'))

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
fq = iq_comp_map.starmap(fq_getter, stream_name='fq', **fq_kwargs)
pdf = iq_comp_map.starmap(pdf_getter, stream_name='pdf', **pdf_kwargs)
