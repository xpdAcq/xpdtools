import operator as op

import numpy as np
from streamz import Stream

raw_foreground = Stream(stream_name='raw foreground')
raw_foreground_dark = Stream(stream_name='raw foreground dark')
raw_background = Stream(stream_name='raw background')
raw_background_dark = Stream(stream_name='raw background dark')
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
img_shape = (bg_corrected_img.map(np.shape).unique(history=1))
