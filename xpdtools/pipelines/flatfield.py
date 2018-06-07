import operator as op

import numpy as np
from streamz_ext import Stream

from xpdtools.pipelines.raw_pipeline import (mean, pol_corrected_img,
                                             geometry,
                                             cal_binner, img_shape,
                                             img_counter)
from xpdtools.tools import move_center

# TODO: fix this so it fires before the mean are produced
# TODO: may need to stop geometry in the start documents?
motors = Stream(stream_name='motor positions')

offset_geometry = motors.combine_latest(geometry, emit_on=0).map(move_center)
geometry.connect(offset_geometry)

bins = (cal_binner
        .combine_latest(img_shape, emit_on=0)
        .starmap(lambda x, y: x.binmap.reshape(y)))

mean_array = (mean
              .map(lambda x: np.concatenate((np.ones(1) * np.nan, mean,
                                             np.ones(1) * np.nan)))
              .zip(img_shape, bins)
              .starmap(lambda x, y, z: np.ones(y) * x[z]))
ff = pol_corrected_img.zip(mean_array).starmap(op.truediv)
total_ff = ff.accumulate(lambda x, state: np.nansum(np.asarray([x, state]),
                                                    axis=0))
ave_ff = total_ff.zip(img_counter).starmap(op.truediv)

'''
How to handle pipeline chunks?

We don't want to duplicate code if possible.
We also don't want to screw up the current raw_pipeline.py system.

This might mean that raw_pipeline.py just has connective tissue which plumbs
data from one piece into another. The namespace for this could get rather 
cluttered though. This would make this easier to put code together which needs
very specific execution times.
'''
