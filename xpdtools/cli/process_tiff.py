import fabio
from ..pipelines.raw_pipeline import *
import pyFAI
import os
from skbeam.io.fit2d import fit2d_save, read_fit2d_msk
from skbeam.io.save_powder_output import save_output


def main(poni_file=None, image_files=None, bg_file=None, mask_file=None,
         polarization=.99,
         edge=20,
         lower_thresh=1.,
         upper_thresh=None,
         alpha=3.):
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
    imgs = None
    filenames = None

    # Modify graph
    # create filename nodes
    filename_node = Stream(stream_name='filename').map(lambda x: os.path.splitext(x)[0])
    # write out mask
    mask.zip_latest(filename_node).sink(lambda x: fit2d_save(x[0], x[1]))
    # write out chi
    mean.zip(q).zip_latest(filename_node).sink(lambda x: save_output(x[1], x[0], x[2], 'Q'))

    pol_corrected_img.args = polarization
    if mask_file:
        tmsk = read_fit2d_msk(mask_file)
    else:
        tmsk = None

    mask.kwargs.update(tmsk=tmsk,
                       edge=edge,
                       lower_thresh=lower_thresh,
                       upper_thresh=upper_thresh,
                       alpha=alpha)

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

    for fn, img in zip(filenames, imgs):
        filename_node.emit(fn)
        if bg is None:
            bg = np.zeros(img.shape)
            dark_corrected_background.emit(bg)
        dark_corrected_foreground.emit(img)
