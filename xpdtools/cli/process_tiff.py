"""Main entry point for processing images to I(Q)"""
import os

import fabio
import fire
import numpy as np
import pyFAI
import tifffile
from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from skbeam.io.save_powder_output import save_output
from streamz_ext import Stream
from streamz_ext.link import link

from xpdtools.pipelines.raw_pipeline import make_pipeline, mask_setting
from xpdtools.pipelines.extra import make_zscore, make_median, make_std

pipeline = link(make_pipeline(),make_median(), make_zscore(), make_std())
mask = pipeline['mask']
q = pipeline['q']
mean = pipeline['mean']
median = pipeline['median']
std = pipeline['std']
z_score = pipeline['z_score']
polarization_array = pipeline['polarization_array']
mask_kwargs = pipeline['create_mask'].kwargs
geometry = pipeline['geometry']
dark_corrected_background = pipeline['dark_corrected_background']
dark_corrected_foreground = pipeline['dark_corrected_foreground']

img_extensions = {'.tiff', '.edf', '.tif'}
# Modify graph
# create filename nodes
filename_source = Stream(stream_name='filename')
filename_node = (filename_source.map(lambda x: os.path.splitext(x)[0]))
# write out mask
mask.combine_latest(filename_node, emit_on=0).sink(
    lambda x: fit2d_save(np.flipud(x[0]), x[1]))
mask.combine_latest(filename_node, emit_on=0).sink(
    lambda x: np.save(x[1] + '_mask.npy', x[0]))

# write out chi
outs = [q, mean, median, std]
out_tup = tuple([[] for _ in outs])
out_sinks = tuple([k.sink(L.append) for k, L in zip(outs, out_tup)])

(mean.zip(q).combine_latest(filename_node, emit_on=0).
 map(lambda l: (*l[0], l[1])).
 sink(lambda x: save_output(x[1], x[0], x[2], 'Q')))
(median.zip(q).combine_latest(filename_node, emit_on=0).
 map(lambda l: (*l[0], l[1])).
 sink(lambda x: save_output(x[1], x[0], x[2] + '_median', 'Q')))
(std.zip(q).combine_latest(filename_node, emit_on=0).
 map(lambda l: (*l[0], l[1])).
 sink(lambda x: save_output(x[1], x[0], x[2] + '_std', 'Q')))
(z_score
 .combine_latest(filename_node, emit_on=0)
 .starsink(lambda img, n: tifffile.imsave(n + '_zscore.tif',
                                          data=img.astype(np.float32))))


def main(poni_file=None, image_files=None, bg_file=None, mask_file=None,
         polarization=.99,
         edge=20,
         lower_thresh=1.,
         upper_thresh=None,
         alpha=3., auto_type='median',
         mask_settings='auto',
         flip_input_mask=True):
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
    flip_input_mask: bool, optional
        If True flip the input mask up down, this helps when using fit2d
        defaults to True.

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
    polarization_array.args = (polarization,)
    if mask_file:
        if mask_file.endswith('.msk'):
            # TODO: may need to flip this?
            tmsk = read_fit2d_msk(mask_file)
        else:
            tmsk = np.load(mask_file)
        if flip_input_mask:
            tmsk = np.flipud(tmsk)
    else:
        tmsk = None

    # update all the kwargs
    mask_kwargs.update(tmsk=tmsk,
                       edge=edge,
                       lower_thresh=lower_thresh,
                       upper_thresh=upper_thresh,
                       alpha=alpha,
                       auto_type=auto_type)
    print(mask_kwargs)
    mask_setting.update(setting=mask_settings)

    # Load calibration
    if poni_file is None:
        poni_file = [f for f in os.listdir('.') if f.endswith('.poni')]
        if len(poni_file) != 1:
            raise RuntimeError("There can only be one poni file")
        else:
            poni_file = poni_file[0]
    geo = pyFAI.load(poni_file)

    bg = None
    img_filenames = None

    if image_files is None:
        img_filenames = [i for i in os.listdir('.') if
                         os.path.splitext(i)[-1] in img_extensions]
        # TODO: Test non tiff files
        if all([f.endswith('.tiff') or f.endswith('.tif') for f in
                img_filenames]):
            imgs = (tifffile.imread(i) for i in img_filenames)
        else:
            imgs = (fabio.open(i).data.astype(float) for i in
                    os.listdir('.') if
                    os.path.splitext(i)[-1] in img_extensions)
    else:
        if isinstance(image_files, str):
            image_files = (image_files,)
            img_filenames = image_files
        imgs = (fabio.open(i).data.astype(float) for i in image_files)

    if bg_file is not None:
        bg = fabio.open(bg_file).data.astype(float)

    for k in out_tup:
        k.clear()

    geometry.emit(geo)

    for i, (fn, img) in enumerate(zip(img_filenames, imgs)):
        filename_source.emit(fn)
        if bg is None:
            bg = np.zeros(img.shape)
        dark_corrected_background.emit(bg)
        dark_corrected_foreground.emit(img)

    return tuple([tuple(x) for x in out_tup])


def run_main():  # pragma: no cover
    # If running from a terminal don't output stuff into lists (too much mem)
    for s in out_sinks:
        s.destroy()
    fire.Fire(main)


if __name__ == '__main__':
    run_main()
