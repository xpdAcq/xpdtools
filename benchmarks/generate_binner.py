from tifffile import imread
from xpdtools.tools import generate_binner, generate_map_bin
from profilehooks import profile
import pyFAI

geo = pyFAI.load('test.poni')
img = imread('test.tiff')


def total(geo, img_shape):
    return generate_binner(*generate_map_bin(geo, img_shape))


bo = profile(total)

binner = bo(geo, img.shape)
binner = bo(geo, img.shape)
# 1 call 1.930
# 2 call 2.675
