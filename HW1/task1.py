import numpy as np
from numpy.linalg import norm
import scipy
import scipy.ndimage as ndi
from math import hypot, floor, sqrt
import Image
import ImageDraw
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imread,figure,subplot,plot,show,bar,hist

def edgeDetection(im):
	sigma = 1.4
#	imdata = np.array(im)
	imdata = np.array(im, dtype = float)
	# image: 494 * 343 ---> but Image(x, y) will be represent in array like Arr[y][x], so Arr will become 343 * 494, so Arr[x][y] will refer to the image(x,y)
#	G = imdata
	G = ndi.filters.gaussian_filter(imdata, sigma)
	

	gradOut = Image.new('L', im.size)
	gradx = np.array(gradOut, dtype = float)   
	grady = np.array(gradOut, dtype = float)   

	kernel_x = [[-1, 0, 1],
				[-2, 0, 2],
				[-1, 0, 1]]

	kernel_y = [[-1, -2, -1],
				[ 0,  0,  0],
				[-1, -2, -1]]

	# np.array will change image from 424 * 343 to 343*424 array
	width = im.size[1] 
	height = im.size[0] 
	print width, height

	for x in range(1, width - 1):
		for y in range(1, height -1):
			px = kernel_x[0][0] * G[x-1][y-1] + kernel_x[0][1] * G[x][y-1] + \
			     kernel_x[0][2] * G[x+1][y-1] + kernel_x[1][0] * G[x-1][y] + \
				 kernel_x[1][1] * G[x][y] + kernel_x[1][2] * G[x+1][y] + \
				 kernel_x[2][0] * G[x-1][y+1] + kernel_x[2][1] * G[x][y+1] + \
				 kernel_x[2][2] * G[x+1][y+1]

			py = kernel_y[0][0] * G[x-1][y-1] + kernel_y[0][1] * G[x][y-1] + \
			     kernel_y[0][2] * G[x+1][y-1] + kernel_y[1][0] * G[x-1][y] + \
				 kernel_y[1][1] * G[x][y] + kernel_y[1][2] * G[x+1][y] + \
				 kernel_y[2][0] * G[x-1][y+1] + kernel_y[2][1] * G[x][y+1] + \
				 kernel_y[2][2] * G[x+1][y+1]
			gradx [x][y] = px
			grady [x][y] = py
	
	sobeloutmag = scipy.hypot(gradx, grady)
	sobeloutmag_n = sobeloutmag / np.max(sobeloutmag)
	mag_sup = scipy.hypot(gradx, grady)
	sobeloutdir = scipy.arctan2(grady, gradx)

	for x in range(width):
		for y in range(height):
		    if (sobeloutdir[x][y]<22.5 and sobeloutdir[x][y]>=0) or \
		       (sobeloutdir[x][y]>=157.5 and sobeloutdir[x][y]<202.5) or \
		       (sobeloutdir[x][y]>=337.5 and sobeloutdir[x][y]<=360):
		        sobeloutdir[x][y]=0
		    elif (sobeloutdir[x][y]>=22.5 and sobeloutdir[x][y]<67.5) or \
		         (sobeloutdir[x][y]>=202.5 and sobeloutdir[x][y]<247.5):
		        sobeloutdir[x][y]=45
		    elif (sobeloutdir[x][y]>=67.5 and sobeloutdir[x][y]<112.5)or \
		         (sobeloutdir[x][y]>=247.5 and sobeloutdir[x][y]<292.5):
		        sobeloutdir[x][y]=90
		    else:
		        sobeloutdir[x][y]=135

	for x in range(1, width-1):
		for y in range(1, height-1):
			if sobeloutdir[x][y]==0:
				if (sobeloutmag[x][y]<=sobeloutmag[x][y+1]) or \
				   (sobeloutmag[x][y]<=sobeloutmag[x][y-1]):
				    mag_sup[x][y]=0
			elif sobeloutdir[x][y]==45:
				if (sobeloutmag[x][y]<=sobeloutmag[x-1][y+1]) or \
				   (sobeloutmag[x][y]<=sobeloutmag[x+1][y-1]):
				    mag_sup[x][y]=0
			elif sobeloutdir[x][y]==90:
				if (sobeloutmag[x][y]<=sobeloutmag[x+1][y]) or \
				   (sobeloutmag[x][y]<=sobeloutmag[x-1][y]):
				    mag_sup[x][y]=0
			else:
				if (sobeloutmag[x][y]<=sobeloutmag[x+1][y+1]) or \
				   (sobeloutmag[x][y]<=sobeloutmag[x-1][y-1]):
				    mag_sup[x][y]=0	
	return	mag_sup / np.max(mag_sup), gradx, grady 

def detectCircles(edge_im, r_min, r_max, quan, useGradient):
	height, width = edge_im.shape
	if r_max == 0:
		r_max = min(width, height) / 2
	if r_min == 0:
		r_min = 10
	listR = []
	listC_all = []
	for r in range (r_min, r_max, int((r_max - r_min) / quan)):
		listX = []
		listX_3 = []
		listX_5 = []
		listX, listX_3, listX_5 = detectCircles_radius(edge_im, r, useGradient)
		listC_all.append(listX)
		listR.append(r)

	print listR 
	print listC_all

	return listR, listC_all 
def detectCircles_radius(edge_im, radius, useGradient):

	v_threshold = 0.0  # 0.0 - 1.0 
	edge_threshold = 1.33 * np.median(edge_im != 0) # 0.0 ~1.0


	height, width = edge_im.shape

	H = np.zeros((height, width))
	candidate_map = Image.new("L", (width, height), "black")

	p_candidate_map = candidate_map.load()


	err = 0.2
#   (x - a)**2 + (y - b)**2 = radius**2	
	for x in xrange(0, width):
		print x
		for y in xrange(0, height):
			weight = 1
			if edge_im[y, x] <= edge_threshold: 
				continue
			p_candidate_map[x, y] = 255
			for a in range(max((x - radius), 0), min((x + radius), width)):
				dx = abs(x - a)
				dy = sqrt(radius**2 - (x - a)**2)
				gx = gradx[y][x]
				gy = grady[y][x]

				b1 = int(y + dy + 0.5) 
				b2 = int(y - dy + 0.5) 

				if (useGradient):
					if(dx == 0 or gx == 0):
						pass
					elif(abs(gy * 1.0 / gx - dy * 1.0 / dx) < err):
						if(a > 0 and b1 < height):
							H[b1][a] += 10
						if(a < width and b2 > 0):
							H[b2][a] += 10
					elif(abs(gy * 1.0 / gx + dy * 1.0 / dx) < err):
						if(a > 0 and b2 > 0):
							H[b2][a] += 10
						if(a < width and b1 < height):
							H[b1][a] += 10
			
				if(b1 < height):
					H[b1][a] += 1
				if(b2 >= 0):
					H[b2][a] += 1

	candidate_map.save('maskImage.png')

	H_3 = np.copy(H)
	H_5 = np.copy(H)

	H_3 = quantization(H_3, 3)
	H_5 = quantization(H_5, 5)
	
	H = ((H - np.min(H)) / (np.max(H) - np.min(H)))
	H_3 = ((H_3 - np.min(H)) / (np.max(H_3) - np.min(H_3)))
	H_5 = ((H_5 - np.min(H)) / (np.max(H_5) - np.min(H_5)))

	H_Img = Image.fromarray(np.uint8(H * 255))
	H_Img.save('HImage_1_noThreshold.png')
	
	H[H < v_threshold] = 0
	H_3[H_3 < v_threshold] = 0
	H_5[H_5 < v_threshold] = 0
	
	H_Img = Image.fromarray(np.uint8(H * 255))
	H_Img.save('HImage_1.png')

	H_Img_3 = Image.fromarray(np.uint8(H_3 * 255))
	H_Img_3.save('HImage_3.png')

	H_Img_5 = Image.fromarray(np.uint8(H_5 * 255))
	H_Img_5.save('HImage_5.png')

	listX = []
	listX_3 = []
	listX_5 = []
	
	j = 0
	for item in H:
		j += 1
		for i in range(len(H[0])):
			if item[i] != 0:
				listX.append((i, j)) 	
	j = 0
	for item in H_3:
		j += 1
		for i in range(len(H_3[0])):
			if item[i] != 0:
				listX_3.append((i, j)) 	

	j = 0
	for item in H_5:
		j += 1
		for i in range(len(H_5[0])):
			if item[i] != 0:
				listX_5.append((i, j)) 	

	return listX, listX_3, listX_5

def plotCircles(im, radius, centers, color):
#	color = 'white'
	draw = ImageDraw.Draw(im)
	for item in centers:
#		print item
		x = item[0]
		y = item[1]
		r = radius
		draw.ellipse((x-r, y-r, x+r, y+r), outline = color)
		im.show()
	return im

def quantization(HoughMap, q):
	
	width = HoughMap.shape[1] 
	height = HoughMap.shape[0] 
#	print width, height
	for y in range(((q - 1) / 2), height - ((q - 1) / 2) + 1, q):
		for x in range(((q - 1) / 2), width - ((q - 1) / 2) + 1, q):
			s = 0
			for i in range(-((q - 1) / 2), ((q - 1) / 2) + 1):
				for j in range(-((q - 1) / 2), ((q - 1) / 2) + 1):
					if((y + i) < height and (x + j) < width):
						s += HoughMap[y + i][x + j]
						if (i != 0 or j != 0):	
							HoughMap[y + i][x + j] = 0
			HoughMap[y][x] = s
	return HoughMap

if __name__=='__main__':
	
	# set radius = -1 to detect all the radius of circle 
	radius = -1 
	fileName = "colorful3.png"
	color = 'white'
	useGradient = True
	# only use for radius == -1
	r_min = 150 
	r_max = 500 
	quan = 2 
	
	img_color = Image.open(fileName)

	img = Image.open(fileName).convert("L")
	img.save('grayOriImg.png')

	print 'start edge detection'
	edge_im , gradx, grady = edgeDetection(img)
	img_edge = Image.fromarray(np.uint8(edge_im * 255))
	img_edge.save('EdgeImage.png')
	print 'end edge detection'

	img_result = img_color.copy() 
	img_result_3 = img_color.copy() 
	img_result_5 = img_color.copy() 
	img_result_all = img_color.copy() 
	img_result.save('img_result_origin.png')

	if radius != -1:
		print 'caculation circles...'
		listX, listX_3, listX_5 = detectCircles_radius(edge_im, radius, useGradient) 
		
		print 'listX, centers: '
		print listX
	
		print 'listX_3, centers: '
		print listX_3
	
		print 'listX_5, centers: '
		print listX_5

		img_result = plotCircles(img_result, radius, listX, color)
		img_result.save('img_result_1.png')
		img_result_3 = plotCircles(img_result_3, radius, listX_3, color)
		img_result_3.save('img_result_3.png')
		img_result_5 = plotCircles(img_result_5, radius, listX_5, color)
		img_result_5.save('img_result_5.png')

	else:
		listR, listC_all = detectCircles(edge_im, r_min ,r_max, quan , useGradient) 
		i = 0
		for r in listR:
			img_result_all = plotCircles(img_result_all, r, listC_all[i], 'white')
			img_result_all.save('img_result_all.png')
			i += 1
