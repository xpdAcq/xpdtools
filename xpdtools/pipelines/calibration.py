from streamz import Stream
from xpdtools.calib import img_calibration
from xpdtools.tools import load_geo, generate_map_bin, map_to_binner

from streamz_ext import create_streamz_graph


def make_pipeline():
    """Make the pipeline for x-ray scattering detector calibration"""
    bg_corrected_img = Stream(stream_name='background corrected img')
    img_shape = Stream(stream_name='img_shape')
    # Calibration management
    wavelength = Stream(stream_name="wavelength")
    calibrant = Stream(stream_name="calibrant")
    detector = Stream(stream_name="detector")
    is_calibration_img = Stream(stream_name="Is Calibration")
    geo_input = Stream(stream_name="geo_input")
    gated_cal = (
        bg_corrected_img.combine_latest(is_calibration_img, emit_on=0)
        .filter(lambda a: bool(a[1]))
        .pluck(0, stream_name="Gate calibration")
    )
    gen_geo_cal = gated_cal.combine_latest(
        wavelength, calibrant, detector, emit_on=0, stream_name='gen_geo_cal'
    ).starmap(img_calibration)
    gen_geo = gen_geo_cal.pluck(1, stream_name='gen_geo')
    geometry = (
        geo_input.combine_latest(is_calibration_img, emit_on=0)
        .filter(lambda a: not bool(a[1]))
        .pluck(0, stream_name="Gate calibration")
        .map(load_geo)
        .union(gen_geo, stream_name="geometry")
    )
    geometry_img_shape = geometry.zip_latest(img_shape, stream_name='geometry_img_shape')

    unique_geo = Stream(stream_name='unique_geo')
    unique_geo.sink(lambda x: geometry_img_shape.lossless_buffer.clear())

    # Only create map and bins (which is expensive) when needed
    # (new calibration)
    map_res = geometry_img_shape.starmap(generate_map_bin, stream_name='map_res')
    cal_binner = map_res.starmap(map_to_binner, stream_name='cal_binner')
    return create_streamz_graph(bg_corrected_img)
