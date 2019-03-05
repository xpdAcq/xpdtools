import numpy as np
import tomopy
from numpy.testing import assert_array_almost_equal
import pytest
from xpdtools.pipelines.tomo import recon_wrapper


@pytest.mark.parametrize('n', [2, 3, 4])
def test_recon_wrapper(n):
    projection = np.random.random([10]*n)
    theta = np.linspace(0, 180, 10)
    center = projection.shape[-1] / 2

    res = recon_wrapper(projection, theta, center,
                        algorithm='gridrec')
    assert res is not None
    assert res.shape == projection.shape
    if len(projection.shape) == 3:
        assert_array_almost_equal(res, tomopy.recon(
            projection, theta, center, algorithm='gridrec'))

