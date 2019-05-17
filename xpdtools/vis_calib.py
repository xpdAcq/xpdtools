from xpdtools.calib4 import findrings
import numpy as np
import tifffile as tf
from pkg_resources import resource_filename as rs_fn
import os
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

DATA_DIR=rs_fn("xpdtools", "data/")
filename=["Ni_onTape_forSlimes_andWahsers_20180812-234250_fe70b9_0001.tiff", 
'Ni_calib_20180811-191034_63e554_0001.tiff', 'Ni_pin_20181101-075909_973de2_0001.tiff', 
'Ni_calib_20180920-230956_0eedc4_0001.tiff','Ni_calib_20180923-133223_c2a848_0001.tiff', 
'Ni_20180922-001850_2a1c3b_0001.tiff', 'Ni_cryostream_bracket_20190402-220917_229125_0001_dark_corrected_img.tiff', 
'sub_20170802-212828_Ni_LongSoham_start_ct_30_3bda69_0001.tiff']
impath = os.path.join(DATA_DIR, filename[0])
imarray = tf.imread(impath)
points_image, center_pt=findrings(imarray)

new_img = np.empty((imarray.shape[0], imarray.shape[1], 4))
max_img = np.amax(imarray)
min_img = np.amin(imarray)
for i, row in enumerate(new_img):
    for j, elem in enumerate(row):
        old_val = imarray[i][j]
        scaled = (old_val - min_img) / (max_img - min_img)
        new_img[i][j] = np.array(
                [1.0 - scaled, 1.0 - scaled, 1.0 - scaled, 1.0]
            )

for point_image in points_image:
    for i in range(-2, 2):
        for j in range(-2, 2):
            new_img[point_image[0] + i][point_image[1] + j] = np.array(
                    [1.0, 0.0, 0.0, 1.0]
                )

for i in range(-2, 2):
    for j in range(-2, 2):
        new_img[center_pt[0] + i][center_pt[1] + j] = np.array(
                [1.0, 0.0, 0.0, 1.0]
            )

plt.imshow(new_img)
plt.show()
