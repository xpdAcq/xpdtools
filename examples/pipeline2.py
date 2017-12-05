from dask.dot import dot_graph
from itertools import zip_longest


def get_background(): pass


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
     # 'vis': (vis,),
     'dark': (get_dark, 'raw'),
     'dark_corrected': (sub, 'raw', 'dark'),
     'calibration': (get_calibration, 'raw'),
     'polarization_corrected': (pol_correct, 'calibration',
                                'dark_corrected'),
     # '2dvis': (vis2d, 'dark_corrected', 'vis'),
     'mask': (make_mask, 'calibration', 'polarization_corrected'),
     'iq': (
     integrate, 'polarization_corrected', 'calibration', 'mask'),
     # 'vis1d_iq': (vis1d, 'iq', 'vis'),
     # 'bg_iq': (get_background, 'raw'),
     # 'bg_corrected_iq': (sub, 'iq', 'muxed_bg'),
     # 'gr': (get_gr, 'bg_corrected_iq', 'raw'),
     # 'vis1d_gr': (vis1d, 'gr', 'vis'),
     # 'candidate_structures': (get_candidates, 'raw'),
     # 'fit_structures': (fit, 'candidate_structures', 'gr'),
     # 'vis_struc': (vis_struct, 'fit_structures', 'vis')
     }

dot_graph(t, 'xpd_pipeline2.pdf')
