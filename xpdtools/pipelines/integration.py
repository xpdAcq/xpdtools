import numpy as np
from skbeam.core.utils import q_to_twotheta
from streamz_ext import Stream

# TODO: break this up maybe?
# This would allow users more control if they didn't want tth or Q
from xpdtools.tools import map_to_binner


def binner_pipeline():
    map_res = Stream()
    mask = Stream()
    pol_corrected_img = Stream()
    wavelength = Stream()

    binner = (
        map_res
        .combine_latest(mask, emit_on=1)
        .map(lambda x: (x[0][0], x[0][1], x[1]))
        .starmap(map_to_binner))
    f_img_binner = pol_corrected_img.map(np.ravel).combine_latest(binner,
                                                                  emit_on=0)
    q = binner.map(getattr, 'bin_centers', stream_name='Q')
    tth = (
        q.combine_latest(wavelength, emit_on=0)
        .starmap(q_to_twotheta, stream_name='tth').map(np.rad2deg))

    mean = (
        f_img_binner.
        starmap(lambda img, binner, **kwargs: binner(img, **kwargs),
                statistic='mean', stream_name='Mean IQ').map(np.nan_to_num))
    return {'map_res': map_res,
            'mask': mask,
            'pol_corrected_img': pol_corrected_img,
            'wavelength': wavelength,
            'binner': binner,
            'f_img_binner': f_img_binner,
            'q': q,
            'tth': tth,
            'mean': mean,
            }
