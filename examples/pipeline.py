from dask.dot import dot_graph
from itertools import zip_longest


def get_background(): pass


def remux(real_stream, track_stream):
    s1, s2 = [next(s) for s in [real_stream, track_stream]]
    yield s1
    s1, s2 = [next(s) for s in [real_stream, track_stream]]
    yield s1
    n, canonical_event = next(real_stream)
    for n, d in track_stream:
        if n == 'stop':
            break
        yield n, canonical_event

    yield from real_stream


def sub(): pass


def get_calibration(): pass


def pol_correct(): pass


def make_mask(): pass


def get_dark(): pass


def integrate(): pass


def get_gr(): pass


def get_raw(): pass


def vis2d(): pass


def vis(): pass


def vis1d(): pass


def get_candidates(): pass


def fit(): pass


def vis_struct(): pass


t = {'raw': (get_raw,),
     'vis': (vis,),
     'dark': (get_dark, 'raw'),
     'muxed_dark': (remux, 'dark', 'raw'),
     'dark_corrected': (sub, 'raw', 'muxed_dark'),
     'calibration': (get_calibration, 'raw'),
     'muxed_calibration': (remux, 'calibration', 'raw'),
     'polarization_corrected': (pol_correct, 'muxed_calibration',
                                'dark_corrected'),
     '2dvis': (vis2d, 'dark_corrected', 'vis'),
     'mask': (make_mask, 'muxed_calibration', 'polarization_corrected'),
     'muxed_mask': (remux, 'mask', 'raw'),
     'iq': (
     integrate, 'polarization_corrected', 'muxed_calibration', 'muxed_mask'),
     'vis1d_iq': (vis1d, 'iq', 'vis'),
     'bg_iq': (get_background, 'raw'),
     'muxed_bg': (remux, 'bg_iq', 'raw'),
     'bg_corrected_iq': (sub, 'iq', 'muxed_bg'),
     'gr': (get_gr, 'bg_corrected_iq', 'raw'),
     'vis1d_gr': (vis1d, 'gr', 'vis'),
     # 'candidate_structures': (get_candidates, 'raw'),
     # 'fit_structures': (fit, 'candidate_structures', 'gr'),
     # 'vis_struc': (vis_struct, 'fit_structures', 'vis')
     }

dot_graph(t, 'xpd_pipeline.pdf')
