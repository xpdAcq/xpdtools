from tifffile import imread
from xpdtools.tools import generate_binner
from profilehooks import profile
import pyFAI
geo = pyFAI.load('test.poni')
img = imread('test.tiff')

bo = profile(generate_binner)

binner = bo(geo, img.shape)
binner = bo(geo, img.shape)
# 1 call 1.930
# 2 call 2.675
