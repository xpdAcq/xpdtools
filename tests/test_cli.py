import os

import numpy as np
from numpy.testing import assert_equal

from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from xpdsim import pyfai_poni, image_file
from xpdtools.cli.process_tiff import main


def test_main(tmp_dir):
    poni_file = pyfai_poni
    # Copy the image file to the temp dir
    main(poni_file, image_file)
    files = os.listdir(tmp_dir)
    for ext in ['.msk', '_mask.npy', '.chi', '_median.chi', '_std.chi',
                '_zscore.png']:
        assert 'test' + ext in files


def test_main_fit2d_mask(tmp_dir):
    poni_file = pyfai_poni
    # Copy the image file to the temp dir
    msk = np.random.randint(0, 2, 2048 * 2048, dtype=bool).reshape(
        (2048, 2048))
    fit2d_save(msk, 'mask_test', tmp_dir)

    # Copy the poni and image files to the temp dir
    main(poni_file, image_file, edge=None, lower_thresh=None, alpha=None,
         mask_file=os.path.join(tmp_dir, 'mask_test.msk'))
    files = os.listdir(tmp_dir)
    for ext in ['.msk', '_mask.npy', '.chi', '_median.chi', '_std.chi',
                '_zscore.png']:
        assert 'test' + ext in files
    msk2 = read_fit2d_msk(os.path.join(tmp_dir, 'test.msk'))
    assert_equal(msk, msk2)
