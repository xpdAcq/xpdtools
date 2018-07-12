from streamz_ext import Stream, create_streamz_graph
import operator as op


def make_pipeline():
    """Make pipeline for scattering (polarization) corrections"""
    geometry_img_shape = Stream(stream_name='geometry_img_shape')
    bg_corrected_img = Stream(stream_name="bg_corrected_img")

    polarization_array = geometry_img_shape.starmap(
        lambda geo, shape, polarization_factor: geo.polarization(
            shape, polarization_factor
        ),
        .99, stream_name='polarization_array'
    )
    pol_correction_combine = bg_corrected_img.combine_latest(
        polarization_array, emit_on=bg_corrected_img,
        stream_name='pol_correction_combine'
    )
    pol_corrected_img = pol_correction_combine.starmap(op.truediv,
                                                       stream_name='pol_corrected_img')
    return create_streamz_graph(geometry_img_shape)
