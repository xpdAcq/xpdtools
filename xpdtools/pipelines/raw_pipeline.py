"""Main pipeline for processing images to I(Q) and PDF"""
import operator as op

from skbeam.core.utils import q_to_twotheta
from streamz import Stream

from xpdtools.calib import img_calibration
from xpdtools.tools import *

# Default kwargs
mask_kwargs = {}
fq_kwargs = dict(dataformat='QA', qmaxinst=28, qmax=25, rstep=np.pi / 25)
pdf_kwargs = dict(dataformat='QA', qmaxinst=28, qmax=22, rstep=np.pi / 22)

# Detector corrections
raw_foreground = Stream(stream_name='raw foreground')
raw_foreground_dark = Stream(stream_name='raw foreground dark')
raw_background = Stream(stream_name='raw background')
raw_background_dark = Stream(stream_name='raw background dark')
dark_corrected_foreground = (
    raw_foreground.
        zip_latest(raw_foreground_dark).
        starmap(op.sub))
dark_corrected_background = (
    raw_background.
        zip_latest(raw_background_dark).
        starmap(op.sub))
bg_corrected_img = (
    dark_corrected_foreground.
        zip_latest(dark_corrected_background).
        starmap(op.sub, stream_name='background corrected img')
)

# Calibration management
wavelength = Stream(stream_name='wavelength')
calibrant = Stream(stream_name='calibrant')
detector = Stream(stream_name='detector')
is_calibration_img = Stream(stream_name='Is Calibration')
geo_input = Stream(stream_name='geometry')
gated_cal = (bg_corrected_img.
    zip_latest(is_calibration_img).
    filter(lambda a: bool(a[1])).
    pluck(0, stream_name='Gate calibration'))

gen_geo_cal = (gated_cal.
    zip_latest(wavelength,
               calibrant,
               detector).
    map(img_calibration))

gen_geo = gen_geo_cal.pluck(1)

geometry = (geo_input.zip_latest(is_calibration_img).
    filter(lambda a: not bool(a[1])).
    pluck(0, stream_name='Gate calibration').
    map(load_geo).
    union(gen_geo, stream_name='Combine gen and load cal'))

# Image corrections
pol_corrected_img = (bg_corrected_img.
    zip_latest(geometry).
    starmap(polarization_correction, .99, stream_name='corrected image')
)

mask = (pol_corrected_img.
    zip_latest(geometry).
    starmap(mask_img, stream_name='mask', **mask_kwargs))

# Integration
binner = (mask.
    zip_latest(geometry).
    starmap(lambda mask, geo: generate_binner(geo, mask=mask)))
f_img_binner = pol_corrected_img.map(np.ravel).zip_latest(binner)

mean = f_img_binner.starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
                          statistic='mean')
median = f_img_binner.starmap(
    lambda img, binner, **kwargs: binner(img, **kwargs), statistic='median')
std = f_img_binner.starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
                         statistic='std')

q = binner.map(getattr, 'bin_centers')
tth = q.zip_latest(wavelength).starmap(q_to_twotheta, stream_name='tth')

z_score = (pol_corrected_img.zip_latest(binner).starmap(z_score_image,
                                                        stream_name='z score').
    zip_latest(mask).starmap(overlay_mask))

# PDF
composition = Stream(stream_name='composition')
iq_comp = q.zip(mean).zip_latest(composition)
fq = iq_comp.starmap(fq_getter, stream_name='fq', **fq_kwargs)
pdf = iq_comp.starmap(pdf_getter, stream_name='pdf', **pdf_kwargs)
