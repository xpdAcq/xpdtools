from streamz_ext import Stream
import numpy as np
from xpdtools.tools import z_score_image, overlay_mask


def make_median():
    """Make pipeline for median integration"""
    f_img_binner = Stream()
    median = (
        f_img_binner.
            starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
                    statistic='median').map(np.nan_to_num))
    return {
        'f_img_binner': f_img_binner,
        'median': median
    }


def make_std():
    """Make pipeline for standard deviation integration"""
    f_img_binner = Stream()
    std = (
        f_img_binner.
            starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
                    statistic='std').map(np.nan_to_num))
    return {
        'f_img_binner': f_img_binner,
        'std': std
    }


def make_zscore():
    """Make pipeline for image z-scoring"""
    pol_corrected_img, binner, mask = Stream(), Stream(), Stream()
    z_score = (
        pol_corrected_img.
            combine_latest(binner, emit_on=0).
            starmap(z_score_image, stream_name='z score').
            combine_latest(mask, emit_on=0).starmap(overlay_mask))
    return {
        'pol_corrected_img': pol_corrected_img,
        'binner': binner,
        'mask': mask,
        'z_score': z_score
    }
