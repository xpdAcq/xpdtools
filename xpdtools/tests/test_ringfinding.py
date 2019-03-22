from xpdtools.calib4 import findrings

import tifffile as tf

from pkg_resources import resource_filename as rs_fn

import os

DATA_DIR=rs_fn("xpdtools", "data/")

filename="Ni_pin_20181101-075909_973de2_0001.tiff"

impath=os.path.join(DATA_DIR, filename)

imarray=tf.imread(impath)

def test_ringfinding():
	center_pt, pointsimage=findrings(imarray)
	#assert center_pt==right_center