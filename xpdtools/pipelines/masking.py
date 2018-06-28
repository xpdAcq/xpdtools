import numpy as np
from streamz_ext import Stream

from xpdtools.tools import mask_img

mask_setting = {"setting": "auto"}


def make_pipeline():
    """Make pipeline for automatic masking"""
    pol_corrected_img = Stream()
    cal_binner = Stream()
    # emit on img so we don't propagate old image data
    # note that the pol_corrected_img has touched the geometry and so always
    # comes after the geometry itself, so we never have a condition where we
    #  fail to emit because pol_corrected_img comes down first
    img_cal_binner = pol_corrected_img.combine_latest(
        cal_binner, emit_on=pol_corrected_img
    )
    # This emits on every image if auto
    all_mask = img_cal_binner.filter(
        lambda x, **kwargs: mask_setting["setting"] == "auto"
    )
    img_counter = Stream(stream_name="img counter")
    # This emits on only the first image if first
    first_mask = (
        img_cal_binner.filter(
            lambda x, **kwargs: mask_setting["setting"] == "first"
        )
        .zip(img_counter)
        .filter(lambda x: x[1] == 1)
        .pluck(0)
    )
    create_mask = all_mask.union(first_mask).starmap(
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

    no_mask = img_cal_binner.filter(
        lambda x, **kwargs: mask_setting["setting"] == "none"
    ).starmap(lambda img, *_: np.ones(img.shape, dtype=bool))
    mask = create_mask.union(no_mask)
    return {
        "pol_corrected_img": pol_corrected_img,
        "cal_binner": cal_binner,
        "img_cal_binner": img_cal_binner,
        "all_mask": all_mask,
        "img_counter": img_counter,
        "first_mask": first_mask,
        "no_mask": no_mask,
        "mask": mask,
        "create_mask": create_mask,
    }
