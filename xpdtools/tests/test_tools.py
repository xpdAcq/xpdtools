##############################################################################
#
# xpdan            by Billinge Group
#                   Simon J. L. Billinge sb2896@columbia.edu
#                   (c) 2016 trustees of Columbia University in the City of
#                        New York.
#                   All rights reserved
#
# File coded by:    Christopher J. Wright
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
import numpy as np
from numpy.testing import assert_array_equal

from xpdtools.tools import (mask_ring_mean, mask_ring_median, load_geo,
                            generate_binner)
from xpdtools.tests.utils import pyFAI_calib


def test_mask_ring_mean():
    values = np.asarray([0, 0, 0, 10, 0, 0, 0, 0])
    positions = np.arange(0, len(values))
    assert mask_ring_mean(values, positions, 1) == np.argmax(values)


def test_mask_ring_median():
    values = np.asarray([0, 0, 0, 1, 0, 0, 0, 0])
    positions = np.arange(0, len(values))
    assert mask_ring_median(values, positions, 3) == np.argmax(values)


def test_load_geo():
    geo = load_geo(pyFAI_calib)
    assert geo


def test_generate_binner():
    geo = load_geo(pyFAI_calib)
    b = generate_binner(geo, (2048, 2048))
    assert b
