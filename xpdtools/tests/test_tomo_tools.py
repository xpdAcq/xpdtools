import numpy as np
import tomopy
from numpy.testing import assert_array_almost_equal
import pytest
from rapidz import Stream
from xpdtools.pipelines.tomo import recon_wrapper, tomo_stack_2D


@pytest.mark.parametrize("n", [2, 3, 4])
def test_recon_wrapper(n):
    projection = np.random.random([10] * n)
    theta = np.linspace(0, 180, 10)
    center = projection.shape[-1] / 2

    res = recon_wrapper(projection, theta, center, algorithm="gridrec")
    assert res is not None
    assert res.shape == projection.shape
    if len(projection.shape) == 3:
        assert_array_almost_equal(
            res, tomopy.recon(projection, theta, center, algorithm="gridrec")
        )


def test_tomo_stack_2D():
    rec = Stream()
    stack_position = Stream()
    start = Stream()
    ns = tomo_stack_2D(rec, stack_position, start)
    L = ns["rec_3D"].sink_to_list()

    stack_position.emit(0)
    rec.emit(np.ones((10, 10)))
    stack_position.emit(0)
    rec.emit(np.ones((10, 10)) * 2)
    assert_array_almost_equal(L[-1], np.ones((10, 10, 1)) * 2)

    stack_position.emit(1)
    rec.emit(np.ones((10, 10, 1)) * 3)
    assert_array_almost_equal(
        L[-1], np.dstack((np.ones((10, 10, 1)) * 2, np.ones((10, 10, 1)) * 3))
    )
