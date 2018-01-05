from time import time
import numpy as np
from tifffile import imread
import pyFAI
from xpdtools.pipelines.raw_pipeline import *
from profilehooks import profile

mask.kwargs['bs_width'] = None
# dark_corrected_background.sink(print)
# pol_corrected_img_zip.sink(print)
# mask.sink(print)
# binner.sink(print)
# mean.sink(print)

geo = pyFAI.load('test.poni')
img = imread('test.tiff')

geometry.emit(geo)

for n in [raw_background_dark, raw_background, raw_foreground_dark]:
    n.emit(np.zeros(img.shape))


@profile(skip=1, sort='module', entries=1000)
def f():
    raw_foreground.emit(img)


for i in range(2):
    f()
