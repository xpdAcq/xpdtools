"""
Pipelines for integration
"""
import numpy as np
from skbeam.core.utils import q_to_twotheta
from streamz_ext import Stream, create_streamz_graph

# TODO: break this up maybe?
# This would allow users more control if they didn't want tth or Q
from xpdtools.tools import map_to_binner


def make_pipeline():
    """Make pipeline for mean integration"""
    map_res = Stream(stream_name='map_res')
    mask = Stream(stream_name='mask')
    pol_corrected_img = Stream(stream_name='pol_corrected_img')
    wavelength = Stream(stream_name='wavelength')

    binner = (
        map_res
            .combine_latest(mask, emit_on=1)
            .map(lambda x: (x[0][0], x[0][1], x[1]))
            .starmap(map_to_binner, stream_name='binner'))
    f_img_binner = (pol_corrected_img
        .map(np.ravel)
        .combine_latest(binner, emit_on=0, stream_name='f_img_binner'))
    q = binner.map(getattr, "bin_centers", stream_name="q")
    tth = (
        q.combine_latest(wavelength, emit_on=0)
        .starmap(q_to_twotheta)
        .map(np.rad2deg, stream_name="tth")
    )

    mean = f_img_binner.starmap(
        lambda img, binner, **kwargs: binner(img, **kwargs),
        statistic="mean",
    ).map(np.nan_to_num, stream_name="mean")
    return create_streamz_graph(mean)
