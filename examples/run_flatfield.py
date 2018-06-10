import matplotlib.pyplot as plt
import numpy as np
from pyFAI.geometry import Geometry
from xpdtools.tools import generate_binner
from matplotlib.colors import LogNorm, SymLogNorm
from bluesky.callbacks.broker import LiveImage
from bluesky.utils import install_qt_kicker

from xpdtools.pipelines.flatfield import *

install_qt_kicker()

ff = np.ones((2048, 2048))
# ff = np.random.random((2048, 2048))

ff *= np.random.normal(1, .01, size=(2048, 2048))
quad = np.ones((2048, 2048))
quad[1024:, 1024:] *= .5
ff *= quad
ff = np.abs(ff)
mask_setting['setting'] = 'none'
# '''
# '''
k = 5
geo = Geometry(wavelength=.18e-10, detector='perkin', dist=.18,
               poni1=.1024 * 2,
               poni2=.1024 * 2,
               rot1=0, rot2=0, rot3=0)
raw_foreground_dark.emit(0.0)
raw_background_dark.emit(0.0)
raw_background.emit(0.0)

ave_ff.map(lambda x: ((ff / x) - 1) * 100).map(np.nan_to_num).sink(
    LiveImage('image', cmap='viridis',
              limit_func=lambda x: (np.nanpercentile(x, .1),
                                    np.nanpercentile(x, 99.9)),
              #                   norm=SymLogNorm(.1),
              window_title='percent off').update)
(mean_array.map(np.nan_to_num)
 .sink(LiveImage('image', cmap='viridis',
                 window_title='predicted flat field',
                 limit_func=lambda x: (
                     np.nanpercentile(x, .1),
                     np.nanpercentile(x, 99.9))
                 ).update))
raw_foreground.map(np.nan_to_num).sink(LiveImage('image',
                                                 cmap='viridis',
                                                 window_title='w/o flat field',
                                                 limit_func=lambda x: (
                                                     np.nanpercentile(x, .1),
                                                     np.nanpercentile(x, 99.9))
                                                 ).update)
is_calibration_img.emit(False)
geo_input.emit(geo.getPyFAI())

for i, p in enumerate(np.linspace(.2, -.2, k)):
    for j, pp in enumerate(np.linspace(.2, -.2, k)):
        img_counter.emit(i * k + j + 1)
        motors.emit((p, pp))
        new_geo = move_center((p, pp), geo)
        q2 = new_geo.qArray((2048, 2048)) / 10.
        img = np.exp(-q2 / 25) * 10000
        img2 = img * ff
        raw_foreground.emit(img2)
        plt.pause(.1)
plt.show()
