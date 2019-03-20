import numpy as np

import tifffile as tf

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

filename = 'Ni_pin_20181101-075909_973de2_0001.tiff'


imarray=tf.imread(filename)


class Slyce():
	def __init__(self,direction,index,image):
		self.direction=direction
		self.index=index
		self.pixels=[]
		if direction=='v':
			self.data=list(image[:,index])
		if direction=='h':
			self.data=list(image[index,:])

def findcenter(image):
	"""
	Takes numpy array corresponding to a tiff image and finds the pixel indices corresponding to the center of the image (not the center of the rings)

	Parameters: numpy array of tiff image
		image: numpy array of tiff image

	Returns:
		r: integer row index of image center
		c: integer column index of image center

	"""

	s=image.shape
	# number of rows divided by 2
	r=s[0]/2
	# number of columns divided by 2
	c=s[1]/2
	return r,c


def clickpoints(slyce,thresh):
	"""
	Takes a slyce and threshold value and adds to the "pixels" attribute of Slyce class the indices of 1 or 2 points (in the 1-D slyce array) that correspond to points on the inner ring.

	Parameters:
		slyce: 1-D numpy array of image data corresponding to a vertical or horizontal slice through the tiff image
		thresh: np.float64 corresponding to a percentage of the maximum value in image numpy array

	"""
	p1=np.argmax(slyce.data)
	p2=np.argmax(slyce.data[:p1]+slyce.data[p1+1:])

	#if statement to determine if max points are large enough to actually be on center ring. If so, the pixel attribute of Slyce is modified to include these indices
	if slyce.data[p2]>=thresh:
		slyce.pixels.append(p1)
		slyce.pixels.append(p2)

	elif slyce.data[p1]>=thresh:


		slyce.pixels.append(p1)

def finddistance(a,b):
	"""
	Finds distance between two points.

	Parameters:
		a: numpy array, list, or tuple containing a coordinates, where a[0] contains a row index and a[1] contains col index
		b: numpy array, list, or tuple containing a coordinates, where a[0] contains a row index and a[1] contains col index 

	Returns:
		np.float64 corresponding to distance between two points
	"""

   	dif1=a[1]-b[1]
   	dif2=a[0]-b[0]
   	d_sq=np.abs(dif1**2+dif2**2)
   
   	return np.sqrt(d_sq)

def findringcenter(image,thres=0.2,d=20):
	"""
	Takes a numpy array corresponding to a tiff image and finds the pixel index of the center of the inner ring.

	Parameters:
		image: numpy array of tiff image
		d: integer corresponding distance around image center (distinct from inner ring center)
	Returns:
		Pixel indices of an estimate of center point of inner ring. 

	
	"""
	thresh=thres*np.max(image)
	r,c=findcenter(image)
	
	#take 10 slices within a range of 'd' away from the cetner of the image and puts them into a list
	slyce1=Slyce('v',c-d,image)
	slyce2=Slyce('v',c+d,image)
	slyce3=Slyce('v',c,image)
	slyce4=Slyce('v',c-d/2,image)
	slyce5=Slyce('v',c+d/2,image)

	slyce6=Slyce('h',r-d,image)
	slyce7=Slyce('h',r+d,image)
	slyce8=Slyce('h',r,image)
	slyce9=Slyce('h',r+d/2,image)
	slyce10=Slyce('h',r-d/2,image)
	slyces=[slyce1, slyce2, slyce3, slyce4, slyce5, slyce6, slyce7, slyce8, slyce9, slyce10]



	points2click=[]
	for slyce in slyces:
		clickpoints(slyce,thresh)
		if slyce.direction=='v':
			for pixel in slyce.pixels:
				points2click.append([pixel,slyce.index])
		if slyce.direction=='h':
			for pixel in slyce.pixels:
				points2click.append([slyce.index,pixel])

		#coords keeps track of the coordinates that are tested to find the center of the inner ring
		coords=[]
		#spread keeps track of the range of point distances from the tested point to the points that have been identified on the center ring
		spread=[]
		
		#the center of the ring is determined by testing a range of points around the center of the image and calculating the distances
		#between that point and the points that have been identified on the inner ring through the clickpoints() function
		#the point that has the smallest range of distances is identified as the center
		for row in range (int(r-3*d),int(r+3*d)):
		   for col in range (int(c-3*d),int(c+3*d)):
				pointdist=[]
				for point in points2click:
					pointdist.append(finddistance([row,col],point))
				spread.append(max(pointdist)-min(pointdist))
				coords.append([row,col])

	center=coords[spread.index(min(spread))]

	return center


def zerocross(lines):
	"""
	Takes in 1-D numpy array and determines array indices corresponding to points where an adjacent value has a different sign and where the difference between the two values before and after zero-crossing exceeds a certain threshold

	Parameters:
		lines: 1-D numpy array
	Returns:
		z: list of indices in lines that correspond to both a change in sign of adjacent values and significant difference in these values

	"""
	z=[]
	for i in range(1,len(lines)-1):
	   if (np.sign(lines[i]) == -1) and (np.sign(lines[i-1]) in [0,1]):
			if np.abs(lines[i]-lines[i-1])>np.max(lines)/80.0:
				z.append(i)
	return z

def findrings(image):

	"""
	Takes a numpy array representing a tiff image and returns the pixel indices of the center point and points on rings 0,1,2,5
	 
	Parameters:
		image: numpy array of tiff image
	Returns:
		pixel index of point on center ring and pixel indices of points on rings 0,1,2,5

	"""  
	center_pt=findringcenter(image)
	print(center_pt)

	#create horizontal slyces from the center of the inner ring (but use 5 and find the median value to avoid issues with hot pixels)
	cs1=Slyce('h',center_pt[0]-2,image)
	cs2=Slyce('h',center_pt[0]-1,image)
	cs3=Slyce('h',center_pt[0],image)
	cs4=Slyce('h',center_pt[0]+1,image)
	cs5=Slyce('h',center_pt[0]+2,image)

	cs=[cs1,cs2,cs3,cs4,cs5]

	value1=cs1.data
	value2=cs2.data
	value3=cs3.data
	value4=cs4.data
	value5=cs5.data
	value_cs=np.zeros(np.shape(value1)[0])

	#create list of median values in horizontal center slyce to avoid hot pixel values
	for i, blank in enumerate(value_cs):
		value_cs[i]=np.median([value1[i],value2[i],value3[i],value4[i],value5[i]])

	#take half of center slyce from the center of the inner ring out
	halfcs=list(value_cs[(center_pt[1]):])

	#find derivative of values on the slyce
	dx=1
	deriv=list(np.gradient(halfcs,dx))


	rings=zerocross(deriv)
	clickpts=[]
	firstpoint=[center_pt[0],rings[0]+center_pt[1]]

	#tests a series of conditions to determine if there is a peak at a certain index in 1D array
	#tests to make sure that peak is real and not noise/small fluctuation
	if (halfcs[rings[0]]>0.7*np.max(halfcs)) or (halfcs[rings[0]-1]>0.7*np.max(halfcs)) or (halfcs[rings[0]+1]>0.7*np.max(halfcs)):
		clickpts.append(rings[0])
		clickpts.append(rings[1])
		clickpts.append(rings[2])
		clickpts.append(rings[5])
	else:
		clickpts.append(rings[1])
		clickpts.append(rings[2])
		clickpts.append(rings[3])
		clickpts.append(rings[6])


	#find indices of points that shoudl be "clicked" in tiff image array
	points_image=[]
	for clickpt in clickpts:
		points_image.append([center_pt[0],clickpt+center_pt[1]])

	print(points_image)
	print(center_pt)



	return points_image, center_pt

points_image,center_pt=findrings(imarray)


new_img = np.empty((imarray.shape[0],imarray.shape[1],4))
max_img = np.amax(imarray)
min_img = np.amin(imarray)
for i, row in enumerate(new_img):
	for j, elem in enumerate(row):
		old_val = imarray[i][j]
		scaled = (old_val-min_img)/(max_img-min_img)
		new_img[i][j] = np.array([1.0-scaled, 1.0-scaled, 1.0-scaled, 1.0])

for point_image in points_image:
	for i in range(-2,2):
		for j in range(-2,2):
			new_img[point_image[0]+i][point_image[1]+j]=np.array([1.0, 0.0, 0.0, 1.0])

for i in range(-2,2):
	for j in range(-2,2):
		new_img[center_pt[0]+i][center_pt+j]=np.array([1.0, 0.0, 0.0, 1.0])



plt.imshow(new_img)
plt.show()
