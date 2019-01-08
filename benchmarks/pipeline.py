from time import time
import numpy as np
from tifffile import imread
import pyFAI
from xpdtools.pipelines.raw_pipeline import *

# from xpdtools.pipelines.extra import *
from profilehooks import profile
from rapidz.link import link

namespace['mask_setting'].update(setting="first")

namespace = link(*pipeline_order, **namespace)



# dark_corrected_background.sink(print)
# pol_corrected_img_zip.sink(print)
# mask.sink(print)
# binner.sink(print)
# mean.sink(print)

geo = pyFAI.load("test.poni")
img = imread("test.tiff")

namespace['geometry'].emit(geo)
namespace['composition'].emit("Au")
for n in [namespace['raw_background_dark'],
          namespace['raw_background'],
          namespace['raw_foreground_dark']]:
    n.emit(np.zeros(img.shape))


@profile(
    skip=1000,
    sort='tottime',
    entries=20,
)
def f(i):
    namespace['img_counter'].emit(i)
    namespace['raw_foreground'].emit(img)


for i in range(5000):
    f(i + 1)

# all masked 5.358 11 calls skip first
# first masked .572 11 calls skip first
