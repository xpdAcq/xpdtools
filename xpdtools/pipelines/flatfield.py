"""Main pipeline for processing images to I(Q) and PDF"""
import operator as op

import numpy as np
from streamz_ext import Stream

from xpdtools.calib import img_calibration
from xpdtools.tools import (load_geo, mask_img, map_to_binner,
                            generate_map_bin, move_center)

mask_setting = {'setting': 'auto'}

motors = Stream(stream_name='motor positions')
raw_foreground = Stream(stream_name='raw foreground')
raw_foreground_dark = Stream(stream_name='raw foreground dark')
raw_background = Stream(stream_name='raw background')
raw_background_dark = Stream(stream_name='raw background dark')

# Get the image shape for the binner
dark_corrected_foreground = (
    raw_foreground
    .combine_latest(raw_foreground_dark, emit_on=0)
    .starmap(op.sub)
)
dark_corrected_background = (
    raw_background
    .combine_latest(raw_background_dark, emit_on=0)
    .starmap(op.sub)
)
bg_corrected_img = (
    dark_corrected_foreground
    .combine_latest(dark_corrected_background, emit_on=0)
    .starmap(op.sub, stream_name='background corrected img')
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
    gated_cal
    .combine_latest(wavelength,
                    calibrant,
                    detector, emit_on=0)
    .starmap(img_calibration)
)

gen_geo = gen_geo_cal.pluck(1)

pre_geometry = (
    geo_input
    .combine_latest(is_calibration_img, emit_on=0)
    .filter(lambda a: not bool(a[1]))
    .pluck(0, stream_name='Gate calibration')
    .map(load_geo)
    .union(gen_geo, stream_name='Combine gen and load cal')
)
# XXX: This is a critical assumption in the workflow we need to make sure
# that the PONI of a calibration can be moved by just moving the detector
geometry = motors.combine_latest(pre_geometry, emit_on=0).starmap(move_center)

# Image corrections
img_shape = (bg_corrected_img.
             map(np.shape)
             .unique(history=1)
             )
geometry_img_shape = geometry.zip_latest(img_shape)

# Only create map and bins (which is expensive) when needed (new calibration)
map_res = geometry_img_shape.starmap(generate_map_bin)
cal_binner = (map_res.starmap(map_to_binner))

bins = (cal_binner
        .combine_latest(img_shape, emit_on=0, first=img_shape)
        .starmap(lambda x, y: x.binmap.reshape(y)))

polarization_array = (
    geometry_img_shape.
        starmap(lambda geo, shape, polarization_factor: geo.polarization(
        shape, polarization_factor), .99)
)

pol_correction_combine = (
    bg_corrected_img
        .combine_latest(polarization_array, emit_on=bg_corrected_img))
pol_corrected_img = pol_correction_combine.pluck(0)  # .starmap(op.truediv)

# emit on img so we don't propagate old image data
# note that the pol_corrected_img has touched the geometry and so always comes
# after the geometry itself, so we never have a condition where  we fail to
# emit because pol_corrected_img comes down first
img_cal_binner = (
    pol_corrected_img.
        combine_latest(cal_binner,
                       emit_on=pol_corrected_img))

all_mask = (
    img_cal_binner
        .filter(lambda x, **kwargs: mask_setting['setting'] == 'auto')
        .starmap(mask_img, stream_name='mask',
                 **dict(edge=30,
                        lower_thresh=0.0,
                        upper_thresh=None,
                        alpha=3,
                        auto_type='median',
                        tmsk=None))
)
img_counter = Stream(stream_name='img counter')
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
        .starmap(lambda img, *_, **kwargs: None)
)

mask = all_mask.union(first_mask, no_mask)

# Integration
binner = (
    map_res
        .combine_latest(mask, emit_on=1)
        .map(lambda x: (x[0][0], x[0][1], x[1]))
        .starmap(map_to_binner))
q = binner.map(getattr, 'bin_centers', stream_name='Q')
f_img_binner = pol_corrected_img.map(np.ravel).combine_latest(binner,
                                                              emit_on=0)

ave = (
    f_img_binner
    .starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
             statistic='mean', stream_name='Mean IQ')
)

mean_array = (ave
              .map(lambda x: np.hstack((np.ones(1) * np.nan, x,
                                        np.ones(1) * np.nan)))
              .combine_latest(img_shape, bins, emit_on=0)
              .starmap(lambda x, y, z: np.ones(y) * x[z]))

calc_ff = pol_corrected_img.zip(mean_array).starmap(op.truediv)
total_ff = calc_ff.accumulate(lambda state, x: state + x)
ave_ff = total_ff.zip(img_counter).starmap(op.truediv)

mask_kwargs = all_mask.kwargs
first_mask.kwargs = mask_kwargs
no_mask.kwargs = mask_kwargs
