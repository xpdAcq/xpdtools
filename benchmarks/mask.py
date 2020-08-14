from tifffile import imread
from xpdtools.tools import binned_outlier, map_to_binner
from profilehooks import profile
import pyFAI

geo = pyFAI.load("test.poni")
img = imread("test.tiff")

bo = profile(binned_outlier, skip=1)
# bo = binned_outlier

binner = map_to_binner(geo, img.shape)
a = binner.argsort_index
b = binner.flatcount

for i in range(2):
    bo(
        img,
        binner,
        # bs_width=None,
        mask_method="mean",
    )

# Median
# binned outlier
# median by itself .714
# median multithread .494
# numba median .270
# multithread numba .178

# mask_img
# numba .336
# regular .520

# Mean
# binned outlier
# numba 4.869
# python 7.351
# numba multithread 2.070
