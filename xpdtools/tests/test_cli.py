import os
import shutil

import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_array_equal
import pytest

from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from xpdsim import pyfai_poni, image_file
from xpdtools.cli.process_tiff import main


def test_main(tmpdir):
    poni_file = pyfai_poni
    dest_image_file = str(tmpdir.join('test.tiff'))
    shutil.copy(image_file, dest_image_file)
    main(poni_file, dest_image_file)
    files = os.listdir(str(tmpdir))
    for ext in ['.msk', '_mask.npy', '.chi', '_median.chi', '_std.chi',
                '_zscore.png']:
        assert 'test' + ext in files


def test_main_fit2d_mask(tmpdir):
    poni_file = pyfai_poni
    # Copy the image file to the temp dir
    msk = np.random.randint(0, 2, 2048 * 2048, dtype=bool).reshape(
        (2048, 2048))
    fit2d_save(msk, 'mask_test', str(tmpdir))
    dest_image_file = str(tmpdir.join('test.tiff'))
    shutil.copy(image_file, dest_image_file)

    # Copy the poni and image files to the temp dir
    main(poni_file, dest_image_file, edge=None, lower_thresh=None, alpha=None,
         mask_file=os.path.join(str(tmpdir), 'mask_test.msk'),
         flip_input_mask=True)
    files = os.listdir(str(tmpdir))
    for ext in ['.msk', '_mask.npy', '.chi', '_median.chi', '_std.chi',
                '_zscore.png']:
        assert 'test' + ext in files
    msk2 = read_fit2d_msk(os.path.join(str(tmpdir), 'test.msk'))
    assert_equal(msk, msk2)


def test_main_no_img(tmpdir):
    poni_file = pyfai_poni
    dest_image_file = str(tmpdir.join('test.tiff'))
    shutil.copy(image_file, dest_image_file)
    os.chdir(str(tmpdir))
    main(poni_file)
    files = os.listdir(str(tmpdir))
    for ext in ['.msk', '_mask.npy', '.chi', '_median.chi', '_std.chi',
                '_zscore.png']:
        assert 'test' + ext in files


keys = ['polarization', 'edge', 'lower_thresh', 'upper_thresh', 'alpha',
        'auto_type']
values = [-.99, 50, 100., 100., 5., 'mean']


@pytest.mark.parametrize(('key', 'value'), zip(keys, values))
def test_main_kwargs(tmpdir, key, value):
    poni_file = pyfai_poni
    dest_image_file = str(tmpdir.join('test.tiff'))
    shutil.copy(image_file, dest_image_file)
    os.chdir(str(tmpdir))
    a = main(poni_file)
    b = main(poni_file, **{key: value})
    assert_raises(AssertionError, assert_array_equal, a[1][0], b[1][0])


def test_main_no_poni(tmpdir):
    poni_file = pyfai_poni
    for file, name in zip([image_file, poni_file], ['test.tiff', 'test.poni']):
        dest_image_file = str(tmpdir.join(name))
        shutil.copy(file, dest_image_file)
    os.chdir(str(tmpdir))
    main()
    files = os.listdir(str(tmpdir))
    for ext in ['.msk', '_mask.npy', '.chi', '_median.chi', '_std.chi',
                '_zscore.png']:
        assert 'test' + ext in files


def test_main_multi_poni(tmpdir):
    poni_file = pyfai_poni
    for file, name in zip([image_file, poni_file, poni_file],
                          ['test.tiff', 'test.poni', 'test2.poni']):
        dest_image_file = str(tmpdir.join(name))
        shutil.copy(file, dest_image_file)
    os.chdir(str(tmpdir))
    print(os.listdir('.'))
    with pytest.raises(RuntimeError):
        main()
