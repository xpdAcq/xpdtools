import numpy as np
import tifffile
import pyFAI

from xpdsim import pyfai_poni, image_file
from xpdtools.pipelines.raw_pipeline import (pdf, composition)
from xpdtools.pipelines.calibration import geometry
from xpdtools.pipelines.image_preprocess import raw_foreground, \
    raw_foreground_dark, raw_background, raw_background_dark

img = tifffile.imread(image_file)
geo = pyFAI.load(pyfai_poni)


def test_raw_pipeline():
    sl = pdf.sink_to_list()
    geometry.emit(geo)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    composition.emit('Au')
    raw_foreground.emit(img)
    assert len(sl) == 1


def test_extra_pipeline():
    from xpdtools.pipelines.extra import z_score
    sl = z_score.sink_to_list()
    geometry.emit(geo)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    raw_foreground.emit(img)
    assert len(sl) == 1


def test_qoi_pipeline():
    from xpdtools.pipelines.qoi import (r_peak_pos, q_peak_pos,
                                        mean_peaks, mean_intensity, pdf_peaks,
                                        pdf_intensity,
                                        pdf_argrelmax_kwargs,
                                        mean_argrelmax_kwargs)
    sls = [k.sink_to_list() for k in [mean_peaks, q_peak_pos,
                                      pdf_peaks, r_peak_pos,
                                      mean_intensity, pdf_intensity, ]]
    geometry.emit(geo)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    composition.emit('Au')
    raw_foreground.emit(img)
    for sl in sls:
        print(sl)
        assert len(sl) == 1
    assert pdf_argrelmax_kwargs == {'order': 5}
    assert mean_argrelmax_kwargs == {'order': 20}


def test_qoi_pipeline2():
    from xpdtools.pipelines.qoi import (r_peak_pos, q_peak_pos,
                                        mean_peaks, mean_intensity, pdf_peaks,
                                        pdf_intensity,
                                        pdf_argrelmax_kwargs,
                                        mean_argrelmax_kwargs)
    sls = [k.sink_to_list() for k in [mean_peaks, q_peak_pos,
                                      pdf_peaks, r_peak_pos,
                                      mean_intensity, pdf_intensity, ]]
    geometry.emit(geo)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    composition.emit('Au')
    pdf_argrelmax_kwargs.update({'order': 100})
    raw_foreground.emit(img)
    for sl in sls:
        print(sl)
        assert len(sl) == 1
    assert pdf_argrelmax_kwargs == {'order': 100}
    assert mean_argrelmax_kwargs == {'order': 20}
