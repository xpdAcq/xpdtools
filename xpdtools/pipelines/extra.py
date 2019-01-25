import operator as op

import numpy as np
from xpdtools.tools import z_score_image, overlay_mask, call_stream_element


def median_gen(f_img_binner, **kwargs):
    median = f_img_binner.starmap(
        call_stream_element, statistic="median", stream_name="Mean IQ"
    ).map(np.nan_to_num)
    return locals()


def std_gen(f_img_binner, mean, **kwargs):
    std = f_img_binner.starmap(
        call_stream_element, statistic="std", stream_name="Mean IQ"
    ).map(op.truediv).map(np.nan_to_num)
    return locals()


def z_score_gen(pol_corrected_img, binner, mask, **kwargs):
    z_score = (
        pol_corrected_img.combine_latest(binner, emit_on=0)
        .starmap(z_score_image, stream_name="z score")
        .combine_latest(mask, emit_on=0)
        .starmap(overlay_mask)
    )
    return locals()
