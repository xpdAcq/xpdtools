from tifffile import imread
from xpdtools.tools import map_to_binner, generate_map_bin
from profilehooks import profile
import pyFAI

geo = pyFAI.load('test.poni')
img = imread('test.tiff')


@profile(
    # skip=1,
    # sort='module',
    entries=20)
def total(geo, img_shape):
    return generate_map_bin(geo, img_shape)


for i in range(10):
    binner = total(geo, img.shape)
# 8.453 for 10 calls of generate_binner and generate_map_bin
# 6.548 for 10 calls of generate_map_bin
