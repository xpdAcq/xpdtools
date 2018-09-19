import os
import shutil

import numpy as np
from numpy.testing import (
    assert_equal,
    assert_raises,
    assert_array_equal,
    assert_allclose,
)
import pytest

from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from xpdsim import pyfai_poni, image_file
from xpdtools.cli.process_tiff import main, make_main

no_output_main = make_main(False)

expected_outputs = tuple(
    [
        ".msk",
        "_mask.npy",
        ".chi",
        "_median.chi",
        "_std.chi",
        # '_zscore.tif'
    ]
)


def test_main(fast_tmpdir):
    print(fast_tmpdir)
    poni_file = pyfai_poni
    dest_image_file = str(os.path.join(fast_tmpdir, "test.tiff"))
    shutil.copy(image_file, dest_image_file)
    no_output_main(poni_file, dest_image_file)
    files = os.listdir(str(fast_tmpdir))
    for ext in expected_outputs:
        assert "test" + ext in files


def test_main2(fast_tmpdir):
    """Test burn in problems associated with data in existing graph

    Some failures associated with this test:
    - Data is already in the graph so when data is emitted into the geometry
    node the graph is triggered, causing the pipeline to try to use
    old data to save into an old dir. This was resolved by emitting on the
    polarization corrected image when combining with the calibrated binner.
    """
    print(fast_tmpdir)
    poni_file = pyfai_poni
    dest_image_file = str(os.path.join(fast_tmpdir, "test.tiff"))
    shutil.copy(image_file, dest_image_file)
    no_output_main(poni_file, dest_image_file)
    files = os.listdir(str(fast_tmpdir))
    for ext in expected_outputs:
        assert "test" + ext in files


def test_main_fit2d_mask(fast_tmpdir):
    poni_file = pyfai_poni
    # Copy the image file to the temp dir
    msk = np.random.randint(0, 2, 2048 * 2048, dtype=bool).reshape(
        (2048, 2048)
    )
    fit2d_save(msk, "mask_test", str(fast_tmpdir))
    dest_image_file = str(os.path.join(fast_tmpdir, "test.tiff"))
    shutil.copy(image_file, dest_image_file)
    print(dest_image_file)
    # Copy the poni and image files to the temp dir
    no_output_main(
        poni_file,
        dest_image_file,
        edge=None,
        lower_thresh=None,
        alpha=None,
        mask_file=os.path.join(str(fast_tmpdir), "mask_test.msk"),
        flip_input_mask=True,
    )
    files = os.listdir(str(fast_tmpdir))
    for ext in expected_outputs:
        assert "test" + ext in files
    msk2 = read_fit2d_msk(os.path.join(str(fast_tmpdir), "test.msk"))
    assert_equal(msk, msk2)


def test_main_no_img(fast_tmpdir):
    poni_file = pyfai_poni
    dest_image_file = str(os.path.join(fast_tmpdir, "test.tiff"))
    shutil.copy(image_file, dest_image_file)
    os.chdir(str(fast_tmpdir))
    no_output_main(poni_file)
    files = os.listdir(str(fast_tmpdir))
    for ext in expected_outputs:
        assert "test" + ext in files


keys = [
    "polarization",
    "edge",
    "lower_thresh",
    "upper_thresh",
    "alpha",
    "auto_type",
]
values = [.99, 50, 100., 100., 4., "mean"]


@pytest.mark.parametrize(("key", "value"), zip(keys, values))
def test_main_kwargs(fast_tmpdir, key, value):
    poni_file = pyfai_poni
    dest_image_file = str(os.path.join(fast_tmpdir, "test.tiff"))
    shutil.copy(image_file, dest_image_file)
    os.chdir(str(fast_tmpdir))
    kwargs = {"alpha": 100, "polarization": -.99}
    print(kwargs)
    a = make_main(True)(poni_file, **kwargs)
    if value == "mean":
        kwargs.update(alpha=5)
    kwargs.update({key: value})
    print(kwargs)
    b = make_main(True)(poni_file, **kwargs)
    assert_raises(AssertionError, assert_array_equal, a[1][0], b[1][0])
    # Make sure we have actual data
    assert_raises(
        AssertionError, assert_allclose, a[1][0], np.zeros(a[1][0].shape)
    )


def test_main_no_poni(fast_tmpdir):
    poni_file = pyfai_poni
    for file, name in zip([image_file, poni_file], ["test.tiff", "test.poni"]):
        dest_image_file = str(os.path.join(fast_tmpdir, name))
        shutil.copy(file, dest_image_file)
    os.chdir(str(fast_tmpdir))
    no_output_main()
    files = os.listdir(str(fast_tmpdir))
    for ext in expected_outputs:
        assert "test" + ext in files


def test_main_multi_poni(fast_tmpdir):
    poni_file = pyfai_poni
    for file, name in zip(
            [image_file, poni_file, poni_file],
            ["test.tiff", "test.poni", "test2.poni"],
    ):
        dest_image_file = str(os.path.join(fast_tmpdir, name))
        shutil.copy(file, dest_image_file)
    os.chdir(str(fast_tmpdir))
    print(os.listdir("."))
    with pytest.raises(RuntimeError):
        no_output_main()


def test_main_background(fast_tmpdir):
    poni_file = pyfai_poni
    dest_image_file = str(os.path.join(fast_tmpdir, "test.tiff"))
    shutil.copy(image_file, dest_image_file)
    out = main(poni_file, dest_image_file, bg_file=dest_image_file)
    files = os.listdir(str(fast_tmpdir))
    assert_array_equal(out[1][0], np.zeros(out[1][0].shape))
    for ext in expected_outputs:
        assert "test" + ext in files
