import operator as op

from xpdtools.pipelines.calibration import geometry_img_shape
from xpdtools.pipelines.image_preprocess import bg_corrected_img

polarization_array = (
    geometry_img_shape.
    starmap(lambda geo, shape, polarization_factor: geo.polarization(
        shape, polarization_factor), .99))
pol_correction_combine = (
    bg_corrected_img
    .combine_latest(polarization_array, emit_on=bg_corrected_img))
pol_corrected_img = pol_correction_combine.starmap(op.truediv)