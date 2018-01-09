from tifffile import imread
from xpdtools.tools import binned_outlier, generate_binner, mask_img
from profilehooks import profile
import pyFAI
geo = pyFAI.load('test.poni')
img = imread('test.tiff')

bo = profile(mask_img)

binner = generate_binner(geo, img.shape)
a = binner.argsort_index
b = binner.flatcount

bo(img, binner, bs_width=None)

# binned outlier
# median by itself .714
# median multithread .494
# numba median .270
# multithread numba .272

# mask_img
# numba .336
# regular .520