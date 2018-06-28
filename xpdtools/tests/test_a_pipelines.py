import numpy as np
import tifffile
import pyFAI

from xpdsim import pyfai_poni, image_file
from streamz_ext.link import link

img = tifffile.imread(image_file)
geo = pyFAI.load(pyfai_poni)


def test_raw_pipeline():
    from xpdtools.pipelines.raw_pipeline import make_pipeline

    pipeline = make_pipeline()
    sl = pipeline["pdf"].sink_to_list()
    pipeline["geometry"].emit(geo)
    for s in [
        pipeline["raw_background_dark"],
        pipeline["raw_background"],
        pipeline["raw_foreground_dark"],
    ]:
        s.emit(np.zeros(img.shape))
    pipeline["composition"].emit("Au")
    pipeline["raw_foreground"].emit(img)
    assert len(sl) == 1


def test_extra_pipeline():
    from xpdtools.pipelines.extra import make_zscore
    from xpdtools.pipelines.raw_pipeline import make_pipeline

    pipeline = link(make_pipeline(), make_zscore())
    sl = pipeline["z_score"].sink_to_list()
    pipeline["geometry"].emit(geo)
    for s in [
        pipeline["raw_background_dark"],
        pipeline["raw_background"],
        pipeline["raw_foreground_dark"],
    ]:
        s.emit(np.zeros(img.shape))
    pipeline["raw_foreground"].emit(img)
    assert len(sl) == 1


# qois are not currently in production nor will they be this cycle

# def test_qoi_pipeline():
#     from xpdtools.pipelines.qoi import
#     from xpdtools.pipelines.raw_pipeline import make_pipeline
#     pipeline = link(make_pipeline(), make_zscore())
#     sls = [k.sink_to_list() for k in [mean_peaks, q_peak_pos,
#                                       pdf_peaks, r_peak_pos,
#                                       mean_intensity, pdf_intensity, ]]
#     pipeline['geometry'].emit(geo)
#     for s in [pipeline['raw_background_dark'], pipeline['raw_background'],
#               pipeline['raw_foreground_dark'], ]:
#         s.emit(np.zeros(img.shape))
#     pipeline['composition'].emit('Au')
#     pipeline['raw_foreground'].emit(img)
#     for sl in sls:
#         print(sl)
#         assert len(sl) == 1
#     assert pdf_argrelmax_kwargs == {'order': 5}
#     assert mean_argrelmax_kwargs == {'order': 20}


# def test_qoi_pipeline2():
#     from xpdtools.pipelines.qoi import (r_peak_pos, q_peak_pos,
#                                         mean_peaks, mean_intensity, pdf_peaks,
#                                         pdf_intensity,
#                                         pdf_argrelmax_kwargs,
#                                         mean_argrelmax_kwargs)
#     sls = [k.sink_to_list() for k in [mean_peaks, q_peak_pos,
#                                       pdf_peaks, r_peak_pos,
#                                       mean_intensity, pdf_intensity, ]]
#     pipeline['geometry'].emit(geo)
#     for s in [pipeline['raw_background_dark'], pipeline['raw_background'],
#               pipeline['raw_foreground_dark'], ]:
#         s.emit(np.zeros(img.shape))
#     pipeline['composition'].emit('Au')
#     pdf_argrelmax_kwargs.update({'order': 100})
#     pipeline['raw_foreground'].emit(img)
#     for sl in sls:
#         print(sl)
#         assert len(sl) == 1
#     assert pdf_argrelmax_kwargs == {'order': 100}
#     assert mean_argrelmax_kwargs == {'order': 20}
