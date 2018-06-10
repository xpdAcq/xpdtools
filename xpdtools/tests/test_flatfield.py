import numpy as np
from pyFAI.geometry import Geometry

from xpdtools.pipelines.flatfield import (ave_ff, mask_setting,
                                          raw_foreground_dark,
                                          raw_background_dark, raw_background,
                                          is_calibration_img, geo_input,
                                          motors, img_counter, raw_foreground)


def test_flatfield_pipeline():
    L = ave_ff.sink_to_list()

    mask_setting['setting'] = 'none'
    ff = np.ones((2048, 2048))
    # ff = np.random.random((2048, 2048))

    ff *= np.random.normal(1, .01, size=(2048, 2048))

    geo = Geometry(wavelength=.18e-10, detector='perkin', dist=.18,
                   poni1=.1024 * 2,
                   poni2=.1024 * 2,
                   rot1=0, rot2=0, rot3=0)

    raw_foreground_dark.emit(0.0)
    raw_background_dark.emit(0.0)
    raw_background.emit(0.0)
    is_calibration_img.emit(False)
    geo_input.emit(geo.getPyFAI())
    motors.emit((0, 0))
    img_counter.emit(1)

    q2 = geo.qArray((2048, 2048)) / 10.
    img = np.exp(-q2 / 25) * 10000
    img2 = img * ff

    raw_foreground.emit(img2)
    assert len(L) == 1
