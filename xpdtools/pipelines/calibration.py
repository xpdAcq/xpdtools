from streamz import Stream
from xpdtools.calib import img_calibration
from xpdtools.pipelines.image_preprocess import bg_corrected_img, img_shape
from xpdtools.tools import load_geo

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