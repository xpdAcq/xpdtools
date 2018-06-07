"""Main pipeline for processing images to I(Q) and PDF"""

import numpy as np
from skbeam.core.utils import q_to_twotheta
from streamz_ext import Stream

from xpdtools.pipelines.calibration import wavelength, geometry_img_shape
from xpdtools.pipelines.scattering_correction import pol_corrected_img
from xpdtools.tools import (mask_img, map_to_binner,
                            fq_getter, pdf_getter, sq_getter, generate_map_bin)

mask_setting = {'setting': 'auto'}
# Get the image shape for the binner

# Only create map and bins (which is expensive) when needed (new calibration)
map_res = geometry_img_shape.starmap(generate_map_bin)
cal_binner = (map_res.starmap(map_to_binner))

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
    .starmap(lambda img, *_: np.ones(img.shape, dtype=bool))
)

mask = all_mask.union(first_mask, no_mask)

# Integration
binner = (
    map_res
    .combine_latest(mask, emit_on=1)
    .map(lambda x: (x[0][0], x[0][1], x[1]))
    .starmap(map_to_binner))
q = binner.map(getattr, 'bin_centers', stream_name='Q')
tth = (
    q.combine_latest(wavelength, emit_on=0)
    .starmap(q_to_twotheta, stream_name='tth').map(np.rad2deg))

f_img_binner = pol_corrected_img.map(np.ravel).combine_latest(binner,
                                                              emit_on=0)

mean = (
    f_img_binner.
    starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
            statistic='mean', stream_name='Mean IQ').map(np.nan_to_num))

# PDF
composition = Stream(stream_name='composition')
iq_comp = (
    q.combine_latest(mean, emit_on=1)
    .combine_latest(composition, emit_on=0))
iq_comp_map = (iq_comp.map(lambda x: (x[0][0], x[0][1], x[1])))

# TODO: split these all up into their components ((r, pdf), (q, fq)...)
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
