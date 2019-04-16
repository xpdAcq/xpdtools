from xpdtools.calib4 import findrings
import numpy as np

import tifffile as tf

import pyFAI

from pkg_resources import resource_filename as rs_fn

import os
import pytest

DATA_DIR=rs_fn("xpdtools", "data/")

filename=["Ni_onTape_forSlimes_andWahsers_20180812-234250_fe70b9_0001.tiff", 'Ni_calib_20180811-191034_63e554_0001.tiff', 'Ni_pin_20181101-075909_973de2_0001.tiff', 'Ni_calib_20180920-230956_0eedc4_0001.tiff','Ni_calib_20180923-133223_c2a848_0001.tiff', 'Ni_20180922-001850_2a1c3b_0001.tiff']


@pytest.mark.parametrize("filename,poni", [(filename[0], os.path.splitext(filename[0])[0]+'.edf'),
	                                 (filename[1], os.path.splitext(filename[1])[0]+'.edf'),
	                                 (filename[2], os.path.splitext(filename[2])[0]+'.edf'),
	                                 (filename[3], os.path.splitext(filename[3])[0]+'.edf'),
	                                 (filename[4], os.path.splitext(filename[4])[0]+'.edf'),(filename[5], os.path.splitext(filename[5])[0]+'.edf')  ])
def test_ringfinding(filename, poni):
	impath=os.path.join(DATA_DIR, filename)
	imarray = tf.imread(impath)
	pointsimage, center_pt=findrings(imarray)
	d=pyFAI.load(poni)
	centerx=d.getFit2D()['centerX']
	centery=d.getFit2D()['centerY']
	assert abs(center_pt[1]-centerx)<=8 and abs(center_pt[0]-centery)<=8

def test_2Darray():
	with pytest.raises(IndexError):
		findrings(np.random.rand(2048))

@pytest.mark.parametrize("wrong_input",[[1,2,3],3,2.7,(2,3),'banana'])
def test_inputtype(wrong_input):
	with pytest.raises(RuntimeError):
		findrings(wrong_input)

