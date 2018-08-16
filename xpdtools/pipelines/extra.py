import numpy as np
from xpdtools.pipelines.raw_pipeline import (
    f_img_binner,
    pol_corrected_img,
    binner,
    mask,
)
from xpdtools.tools import z_score_image, overlay_mask, call_stream_element

median = f_img_binner.starmap(call_stream_element,
                              statistic="median",
                              stream_name="Mean IQ",
                              ).map(np.nan_to_num)
std = f_img_binner.starmap(call_stream_element,
                           statistic="std",
                           stream_name="Mean IQ",
                           ).map(np.nan_to_num)
z_score = (
    pol_corrected_img.combine_latest(binner, emit_on=0)
        .starmap(z_score_image, stream_name="z score")
        .combine_latest(mask, emit_on=0)
        .starmap(overlay_mask)
)
