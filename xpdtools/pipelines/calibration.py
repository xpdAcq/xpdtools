from streamz import Stream
from xpdtools.calib import img_calibration
from xpdtools.tools import load_geo, generate_map_bin, map_to_binner


def make_pipeline():
    """Make the pipeline for x-ray scattering detector calibration"""
    bg_corrected_img = Stream()
    img_shape = Stream()
    # Calibration management
    wavelength = Stream(stream_name='wavelength')
    calibrant = Stream(stream_name='calibrant')
    detector = Stream(stream_name='detector')
    is_calibration_img = Stream(stream_name='Is Calibration')
    geo_input = Stream(stream_name='geometry')
    gated_cal = (
        bg_corrected_img.
        combine_latest(is_calibration_img, emit_on=0).
        filter(lambda a: bool(a[1])).
        pluck(0, stream_name='Gate calibration'))
    gen_geo_cal = (
        gated_cal.
        combine_latest(wavelength,
                       calibrant,
                       detector, emit_on=0).
        starmap(img_calibration)
    )
    gen_geo = gen_geo_cal.pluck(1)
    geometry = (
        geo_input.combine_latest(is_calibration_img, emit_on=0).
        filter(lambda a: not bool(a[1])).
        pluck(0, stream_name='Gate calibration').
        map(load_geo).
        union(gen_geo, stream_name='Combine gen and load cal'))
    geometry_img_shape = geometry.zip_latest(img_shape)
    # Only create map and bins (which is expensive) when needed
    # (new calibration)
    map_res = geometry_img_shape.starmap(generate_map_bin)
    cal_binner = (map_res.starmap(map_to_binner))
    return {'bg_corrected_img': bg_corrected_img, 'img_shape': img_shape,
            'wavelength': wavelength, 'calibrant': calibrant,
            'detector': detector, 'is_calibration_img': is_calibration_img,
            'geo_input': geo_input, 'gated_cal': gated_cal,
            'gen_geo_cal': gen_geo_cal, 'gen_geo': gen_geo,
            'geometry': geometry, 'geometry_img_shape': geometry_img_shape,
            'map_res': map_res,
            'cal_binner': cal_binner}
