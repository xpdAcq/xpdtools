from streamz_ext import Stream
import operator as op


def make_pipeline():
    """Make pipeline for scattering (polarization) corrections"""
    geometry_img_shape = Stream()
    bg_corrected_img = Stream()

    polarization_array = geometry_img_shape.starmap(
        lambda geo, shape, polarization_factor: geo.polarization(
            shape, polarization_factor
        ),
        .99,
    )
    pol_correction_combine = bg_corrected_img.combine_latest(
        polarization_array, emit_on=bg_corrected_img
    )
    pol_corrected_img = pol_correction_combine.starmap(op.truediv)
    return {
        "geometry_img_shape": geometry_img_shape,
        "bg_corrected_img": bg_corrected_img,
        "polarization_array": polarization_array,
        "pol_correction_combine": pol_correction_combine,
        "pol_corrected_img": pol_corrected_img,
    }
