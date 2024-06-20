
import numpy as np

import tifffile as tf
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


filename = 'Ni_pin_20181101-075909_973de2_0001.tiff'

imarray=tf.imread(filename)
print(imarray)

class Slyce():
	def __init__(self,direction,index,imarray):
		self.direction=direction
		self.index=index
		self.pixels=[]
		if direction=='v':
			self.data=list(imarray[:,index])
		if direction=='h':
			self.data=list(imarray[index,:])

def findringcenter(image,thres=.20,d=20):
	
	def findcenter(image):
		s=image.shape
		# number of rows divided by 2
		r=s[0]/2
		# number of columns divided by 2
		c=s[1]/2
		return r,c

	r,c=findcenter(image)

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

	def findthreshold(image,thres):
		return thres*np.max(image)

	thresh=findthreshold(image,thres)
	
	def clickpoints(slyce,thresh):
		p1=np.argmax(slyce.data)
		p2=np.argmax(slyce.data[:p1]+slyce.data[p1+1:])

		if slyce.data[p2]>=thresh:
			slyce.pixels.append(p1)
			slyce.pixels.append(p2)
		
		elif slyce.data[p1]>=thresh:

			slyce.pixels.append(p1)
	
	points2click=[]
	for slyce in slyces:
		clickpoints(slyce,thresh)
		if slyce.direction=='v':
			for pixel in slyce.pixels:
				points2click.append([pixel,slyce.index])
		if slyce.direction=='h':
			for pixel in slyce.pixels:
				points2click.append([slyce.index,pixel])

	highlight=np.mean(image)*.9
	image_new=image
	
	def finddistance(a,b):
		dif1=a[1]-b[1]
		dif2=a[0]-b[0]
		d_sq=np.abs(dif1**2+dif2**2)
		return np.sqrt(d_sq)

	
	#f, ax = plt.subplots(1)
	#ax.imshow(image)
	ring_mask=np.ones_like(image,dtype=bool)
	

	for point in points2click:
		#ax.scatter(point[0],point[1])
		ring_mask[point[0],point[1]]=False
		image_new[point[0],point[1]]=0
	
	coords=[]
	spread=[]
	for row in range (int(r-3*d),int(r+3*d)):
		for col in range (int(c-3*d),int(c+3*d)):
			pointdist=[]
			for point in points2click:
				pointdist.append(finddistance([row,col],point))
			spread.append(max(pointdist)-min(pointdist))
			coords.append([row,col])

	center=coords[spread.index(min(spread))]
	return center
	print(center)

	

#take a horizontal cut out from the center of the ring to the end of the image
#row is constant and column index goes from col index of center to end

def findrings(image):
	center=findringcenter(image)
	cs1=Slyce('h',center[0]-2,image)
	cs2=Slyce('h',center[0]-1,image)
	cs3=Slyce('h',center[0],image)
	cs4=Slyce('h',center[0]+1,image)
	cs5=Slyce('h',center[0]+2,image)

	cs=[cs1,cs2,cs3,cs4,cs5]
	
	value1=cs1.data
	value2=cs2.data
	value3=cs3.data
	value4=cs4.data
	value5=cs5.data
	value_cs=np.zeros(np.shape(value1)[0])
	for i, blank in enumerate(value_cs):
		value_cs[i]=np.median([value1[i],value2[i],value3[i],value4[i],value5[i]])
	

	halfcs=list(value_cs[(center[1]):])

	#find derivative of values on the slyce
	dx=1
	deriv=list(np.gradient(halfcs,dx))
	
	#creates a function that finds the indices of a list where the derivative changes sign
	def zerocross(lines):
		z=[]
		for i in range(1,len(lines)-1):
			if (np.sign(lines[i]) == -1) and (np.sign(lines[i-1]) in [0,1]):
				if np.abs(lines[i]-lines[i-1])>np.max(lines)/80.0:
					z.append(i)
		return z

	
	center=findringcenter(image)
	rings=zerocross(deriv)
	clickpts=[]
	firstpoint=[center[0],rings[0]+center[1]]

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


	
	points_image=[]
	for clickpt in clickpts:
		points_image.append([center[0],clickpt+center[1]])

	print(points_image)

	plt.plot(range(0,len(halfcs)), halfcs)
	plt.scatter(range(0,len(deriv)),deriv)
	plt.plot(deriv)
	plt.plot(np.zeros(len(deriv)))
	for clickpt in clickpts:
		plt.scatter(clickpt,0,color='r')
	plt.show()


	new_img = np.empty((image.shape[0],image.shape[1],4))
	max_img = np.amax(image)
	min_img = np.amin(image)
	for i, row in enumerate(new_img):
		for j, elem in enumerate(row):
			old_val = image[i][j]
			scaled = (old_val-min_img)/(max_img-min_img)
			new_img[i][j] = np.array([1.0-scaled, 1.0-scaled, 1.0-scaled, 1.0])

	for point_image in points_image:
		for i in range(-2,2):
			for j in range(-2,2):
				new_img[point_image[0]+i][point_image[1]+j]=np.array([1.0, 0.0, 0.0, 1.0])

	

	plt.imshow(new_img)
	plt.show()
	return points_image


findrings(imarray)


