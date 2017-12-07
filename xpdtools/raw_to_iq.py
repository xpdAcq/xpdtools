import operator as op
import os

import fabio
import fire
import pyFAI
from pims import ImageSequence
import tifffile
from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from skbeam.io.save_powder_output import save_output
from streamz import Stream

from xpdtools.tools import *
from xpdtools.pipelines.iq_pipeline import conf_iq_pipeline
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

SUPPORTED_FORMATS = ('.tif', '.tiff', '.edf')
VERSION = '0.0.1-alpha'


def main(poni_file=None,
         img_file=None,
         bg_file=None,
         mask_file=None,
         polarization=.99,
         edge=30,
         lower_thresh=0.0,
         upper_thresh=None,
         bs_width=None, tri_offset=13, v_asym=0,
         alpha=2.5,
         auto_type='median',
         bg_scale=1):
    """Run the data processing protocol taking raw images to background
    subtracted I(Q) files.

    The data processing steps included in this protocol are:
    background subtraction, polarization correction, automated masking, and
    pixel resolution integration

    Parameters
    ----------
    poni_file: str or None, optional
        File generated from pyFAI's calibration, if None look in the current
        working directory for the poni file, defaults to None.
    img_file: str or None, optional
        File to process, if None use all the valid files in the directory,
        defaults to None.
    bg_file: str or None, optional
        Background image, if None no background subtraction is performed,
        defaults to None.
    mask_file: str or None, optional
        Mask file to include in the data processing, if None don't use one,
        defaults to None.
    polarization: float, optional
        The polzarization factor to use, defaults to .99, if None do not
        perform polarization correction
    mask_kwargs: dict, optional
        kwargs to the mask creator, see ``xpdtools.tools.mask_img`` for docs,
        defaults to None
    bg_scale: float, optional
        The scale for the background. Defaults to 1.
    """
    fig, ax = plt.subplots()
    mask_kwargs = dict(edge=edge, lower_thresh=lower_thresh,
                       upper_thresh=upper_thresh,
                       bs_width=bs_width, tri_offset=tri_offset, v_asym=v_asym,
                       alpha=alpha, auto_type=auto_type)
    # Load calibration

    if poni_file is None:
        poni_file = [f for f in os.listdir('.') if f.endswith('.poni')]
        if len(poni_file) != 1:
            RuntimeError("There can only be one poni file")
        else:
            poni_file = poni_file[0]
    geo = pyFAI.load(poni_file)

    # Load images
    if img_file:
        img_file = (img_file,)
    else:
        img_file = [f for f in os.listdir('.') if
                    f.endswith(SUPPORTED_FORMATS)]
    if img_file[0].endswith(('.tif', '.tiff')):
        imgs = ImageSequence(img_file)
        img_shape = imgs[0].shape
    else:
        imgs = (fabio.open(i).data.astype(float) for i in img_file)
        img_shape = fabio.open(img_file[0]).data.shape

    # Load background
    if bg_file:
        bg_img = fabio.open(bg_file).data.astype(float)
    else:
        bg_img = np.zeros(img_shape)

    # Load mask
    if mask_file:
        tmsk = read_fit2d_msk(mask_file)
    else:
        tmsk = None
    mask_kwargs['tmsk'] = tmsk

    # def pipeline
    foreground, background, geometry, iq, z_score, mask = conf_iq_pipeline(
        polarization, mask_kwargs, bg_scale)
    start = Stream(stream_name='start')
    fn = start.pluck(1).map(lambda x: os.path.splitext(x)[0])
    start.pluck(0).sink(ax.imshow)
    start.pluck(0).connect(foreground)
    mask_fn = mask.zip_latest(fn)
    # write to disk
    mask_fn.sink(lambda a: fit2d_save(np.flipud(a[0]), a[1]))
    iq.zip_latest(fn).sink(
        lambda a: save_output(a[0][0], a[0][1], a[1], 'Q'))
    z_score_fn = fn.map(lambda x: x + '_zscore.tif')
    (z_score.
        zip_latest(z_score_fn)
        .sink(lambda a: tifffile.imsave(a[1], a[0].astype(np.float32)))
        )
    # z_score.sink(ax.imshow, norm=SymLogNorm(1.))

    # emit into pipeline
    background.emit(bg_img)
    geometry.emit(geo)
    for img, f in zip(imgs, img_file):
        start.emit((img, f))
    plt.show()


def main2():
    fire.Fire(main)


if __name__ == '__main__':
    fire.Fire(main)
