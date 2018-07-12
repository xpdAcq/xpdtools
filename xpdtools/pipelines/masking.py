import numpy as np
from streamz_ext import Stream, create_streamz_graph

from xpdtools.tools import mask_img

mask_setting = {"setting": "auto"}


def make_pipeline():
    """Make pipeline for automatic masking"""
    pol_corrected_img = Stream(stream_name='pol_corrected_img')
    cal_binner = Stream(stream_name='cal_binner')
    # emit on img so we don't propagate old image data
    # note that the pol_corrected_img has touched the geometry and so always
    # comes after the geometry itself, so we never have a condition where we
    #  fail to emit because pol_corrected_img comes down first
    img_cal_binner = pol_corrected_img.combine_latest(
        cal_binner, emit_on=pol_corrected_img, stream_name='img_cal_binner'
    )
    # This emits on every image if auto
    all_mask = img_cal_binner.filter(
        lambda x, **kwargs: mask_setting["setting"] == "auto",
        stream_name='all_mask'
    )
    img_counter = Stream(stream_name="img_counter")
    # This emits on only the first image if first
    first_mask = (
        img_cal_binner.filter(
            lambda x, **kwargs: mask_setting["setting"] == "first"
        )
        .zip(img_counter)
        .filter(lambda x: x[1] == 1)
        .pluck(0, stream_name='first_mask')
    )
    create_mask = all_mask.union(first_mask).starmap(
        mask_img,
        **dict(
            edge=30,
            lower_thresh=0.0,
            upper_thresh=None,
            alpha=3,
            auto_type="median",
            tmsk=None,
        ), stream_name='create_mask'
    )

    no_mask = img_cal_binner.filter(
        lambda x, **kwargs: mask_setting["setting"] == "none"
    ).starmap(lambda img, *_: np.ones(img.shape, dtype=bool), stream_name='no_mask')
    mask = create_mask.union(no_mask, stream_name="mask",)
    return create_streamz_graph(pol_corrected_img)
