from time import time
import numpy as np
from tifffile import imread
import pyFAI
from xpdtools.tools import *
from profilehooks import profile

# dark_corrected_background.sink(print)
# pol_corrected_img.sink(print)
# mask.sink(print)
# binner.sink(print)
# mean.sink(print)

geo = pyFAI.load('test.poni')
img = imread('test.tiff')

binner = generate_binner(geo, img.shape)

@profile(skip=1,
         # sort='module',
         entries=100)
def f(i):
    z_score_image(img, binner)


for i in range(10):
    f(i+1)
