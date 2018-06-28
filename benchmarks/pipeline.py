from tifffile import imread
import pyFAI
from xpdtools.pipelines.image_preprocess import (
    raw_foreground,
    raw_foreground_dark,
    raw_background,
    raw_background_dark,
)
from xpdtools.pipelines.raw_pipeline import *

# from xpdtools.pipelines.extra import *
from profilehooks import profile

mask_setting.update(setting="first")

# dark_corrected_background.sink(print)
# pol_corrected_img_zip.sink(print)
# mask.sink(print)
# binner.sink(print)
# mean.sink(print)

geo = pyFAI.load("test.poni")
img = imread("test.tiff")

geometry.emit(geo)
composition.emit("Au")
for n in [raw_background_dark, raw_background, raw_foreground_dark]:
    n.emit(np.zeros(img.shape))


@profile(
    skip=1,
    # sort='module',
    entries=20,
)
def f(i):
    img_counter.emit(i)
    raw_foreground.emit(img)


for i in range(11):
    f(i + 1)

# all masked 5.358 11 calls skip first
# first masked .572 11 calls skip first
