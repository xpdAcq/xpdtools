from tifffile import imread
from xpdtools.tools import map_to_binner, generate_map_bin
from profilehooks import profile
import pyFAI
from numba import jit
import numpy as np

geo = pyFAI.load('test.poni')
img = imread('test.tiff')

bo = map_to_binner

binner = bo(*generate_map_bin(geo, img.shape))
f = profile(binner.__call__)
a = binner.xy_argsort


@jit(nopython=True, cache=True)
def b(data):
    return np.max(data)


f(img.flatten(), statistic=np.max)

# median
# standard .255
# numba .2
