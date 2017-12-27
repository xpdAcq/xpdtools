"""Not many tests since most of this is GUI code"""
import os
import time

import yaml
from pyFAI.calibration import Calibration

from xpdtools.calib import _save_calib_param
from xpdtools.tests.utils import pyFAI_calib


def test_save_calib_param(tmpdir):
    c = Calibration(**pyFAI_calib)
    print(c)
    fn = os.path.join(tmpdir, 'test_calib.yaml')
    t = time.time()
    _save_calib_param(c, t, fn)
    assert os.path.exists(fn)
    with open(fn, 'r') as f:
        y = yaml.load(f)
    for k in pyFAI_calib:
        assert pyFAI_calib[k] == y[k]
    assert y['time'] == t
