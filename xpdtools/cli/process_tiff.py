"""Main entry point for processing images to I(Q)"""
import os

import fabio
import fire
import numpy as np
import pyFAI
from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from skbeam.io.save_powder_output import save_output
from streamz_ext import Stream
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from ..pipelines.raw_pipeline import (pol_corrected_img, mask, mean, q,
                                      geometry, dark_corrected_foreground,
                                      dark_corrected_background, z_score, std,
                                      median, mask_setting)


def main(poni_file=None, image_files=None, bg_file=None, mask_file=None,
         polarization=.99,
         edge=20,
         lower_thresh=1.,
         upper_thresh=None,
         alpha=3., auto_type='median',
         mask_settings='auto'):
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
    image_files: str or None, optional
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
    edge: int, optional
        The number of pixels from the edge to mask with an edge mask, defaults
        to 20, if None no edge mask used
    lower_thresh: float, optional
        Threshold for lower threshold mask, all pixels with value less than
        this value (after background subtraction if applicable), defaults to 1.
        if None do not apply lower theshold mask
    upper_thresh: float, optional
        Threshold for upper threshold mask, all pixels with value greater than
        this value (after background subtraction if applicable), defaults to
        None if None do not apply upper theshold mask
    alpha: float, optional
        Number of standard deviations away from the ring mean to mask, defaults
        to 3. if None do not apply automated masking
    auto_type : {'median', 'mean'}, optional
        The type of automasking to use, median is faster, mean is more
        accurate. Defaults to 'median'.
    mask_settings: {'auto', 'first', None}, optional
        If auto mask every image, if first only mask first image, if None
        mask no images. Defaults to None

    Returns
    -------
    q_l : list of ndarrays
        The list of q values
    mean_l : list of ndarrays
        The list of mean values
    median_l : list of ndarrays
        The list of median values
    std_l : list of ndarrays
        The list of standard deviation values
    """
    # Load calibration
    if poni_file is None:
        poni_file = [f for f in os.listdir('.') if f.endswith('.poni')]
        if len(poni_file) != 1:
            RuntimeError("There can only be one poni file")
        else:
            poni_file = poni_file[0]
    geo = pyFAI.load(poni_file)
    bg = None
    filenames = None

    # Modify graph
    # create filename nodes
    filename_source = Stream(stream_name='filename')
    filename_node = (filename_source.map(lambda x: os.path.splitext(x)[0]))
    # write out mask
    mask.zip_latest(filename_node).sink(lambda x: fit2d_save(np.flipud(x[0]),
                                                             x[1]))
    mask.zip_latest(filename_node).sink(lambda x: np.save(x[1] + '_mask.npy',
                                                          x[0]))
    # write out chi
    mean_l = mean.sink_to_list()
    median_l = median.sink_to_list()
    std_l = std.sink_to_list()
    q_l = q.sink_to_list()

    (mean.zip(q).zip_latest(filename_node).
        map(lambda l: (*l[0], l[1])).
        sink(lambda x: save_output(x[1], x[0], x[2], 'Q')))
    (median.zip(q).zip_latest(filename_node).
        map(lambda l: (*l[0], l[1])).
        sink(lambda x: save_output(x[1], x[0], x[2] + '_median', 'Q')))
    (std.zip(q).zip_latest(filename_node).
        map(lambda l: (*l[0], l[1])).
        sink(lambda x: save_output(x[1], x[0], x[2] + '_std', 'Q')))
    fig, ax = plt.subplots()
    # write out zscore
    (z_score.map(ax.imshow, norm=SymLogNorm(1.)).map(fig.colorbar).
        zip_latest(filename_node).
        sink(lambda x: fig.savefig(x[1] + '_zscore.png')))

    pol_corrected_img.args = (polarization,)
    if mask_file:
        if mask_file.endswith('.msk'):
            # TODO: may need to flip this?
            tmsk = read_fit2d_msk(mask_file)
        else:
            tmsk = np.load(mask_file)
    else:
        tmsk = None

    mask.kwargs.update(tmsk=tmsk,
                       edge=edge,
                       lower_thresh=lower_thresh,
                       upper_thresh=upper_thresh,
                       alpha=alpha,
                       bs_width=None,
                       auto_type=auto_type)
    mask_setting.update({'setting': mask_settings})
    geometry.emit(geo)
    if image_files is None:
        filenames = os.listdir('.')
        imgs = (fabio.open(i).data.astype(float) for i in os.listdir('.'))
    else:
        if isinstance(image_files, str):
            image_files = (image_files,)
            filenames = image_files
        imgs = (fabio.open(i).data.astype(float) for i in image_files)
    if bg_file is not None:
        bg = fabio.open(bg_file).data.astype(float)

    for i, (fn, img) in enumerate(zip(filenames, imgs)):
        filename_source.emit(fn)
        if bg is None:
            bg = np.zeros(img.shape)
            dark_corrected_background.emit(bg)
        dark_corrected_foreground.emit(img)
    return q_l, mean_l, median_l, std_l


def run_main():
    fire.Fire(main)


if __name__ == '__main__':
    run_main()
