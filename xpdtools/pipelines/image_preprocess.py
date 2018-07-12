import operator as op

import numpy as np
from streamz import Stream


def make_pipeline():
    """Make pipeline for image pre-processing

    Returns
    -------
    dict of Streams:
        The input and output streams
    """
    raw_foreground = Stream(stream_name="raw foreground")
    raw_foreground_dark = Stream(stream_name="raw foreground dark")
    raw_background = Stream(stream_name="raw background")
    raw_background_dark = Stream(stream_name="raw background dark")

    start_docs = Stream()
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        start_docs.map(lambda x: 0.0).connect(s)

    dark_corrected_foreground = raw_foreground.combine_latest(
        raw_foreground_dark, emit_on=0
    ).starmap(op.sub)
    dark_corrected_background = raw_background.combine_latest(
        raw_background_dark, emit_on=0
    ).starmap(op.sub)
    bg_corrected_img = dark_corrected_foreground.combine_latest(
        dark_corrected_background, emit_on=0
    ).starmap(op.sub, stream_name="background corrected img")
    img_shape = bg_corrected_img.map(np.shape).unique(history=1)
    return dict(
        raw_foreground=raw_foreground,
        raw_foreground_dark=raw_foreground_dark,
        raw_background=raw_background,
        raw_background_dark=raw_background_dark,
        dark_corrected_foreground=dark_corrected_foreground,
        dark_corrected_background=dark_corrected_background,
        bg_corrected_img=bg_corrected_img,
        img_shape=img_shape,
        start_docs=start_docs
    )
