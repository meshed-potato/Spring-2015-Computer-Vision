import cv2
import cv
import numpy as np

if __name__ == "__main__" :
	hog = cv2.HOGDescriptor()
	sample = "Starbucks/image4.jpg"
	# sample = "Starbucks/template.gif"
	im = cv2.imread(sample)
	print im.shape
	# print calc_hog(im)
	h = hog.compute(im)
	print h.shape
	print h[1:100]



	
# https://pebbie.wordpress.com/2011/11/10/computing-hog-features-in-opencv-python/
# def calc_hog(im,numorient=9):
#     """
#     calculate integral HOG (Histogram of Orientation Gradient) image (w,h,numorient)
     
#     calc_hog(im, numorient=9)
     
#     returns 
#         Integral HOG image
 
#     params 
#         im : color image
#         numorient : number of orientation bins, default is 9 (-4..4)
     
#     """
#     sz = cv.GetSize(im)
#     gr = cv.CreateImage(sz, 8, 1)
#     gx = cv.CreateImage(sz, 32, 1)
#     gy = cv.CreateImage(sz, 32, 1)
     
#     #convert to grayscale
#     cv.CvtColor(im, gr, cv.CV_BGR2GRAY)
     
#     #calc gradient using sobel
#     cv.Sobel(gr, gx, 1, 0, 3)
#     cv.Sobel(gr, gy, 0, 1, 3)
     
#     #calc initial result
#     hog = np.zeros((sz[1], sz[0], numorient))
#     mid = numorient/2
#     for y in xrange(0, sz[1]-1):
#         for x in xrange(0, sz[0]-1):
#             angle = int(round(mid*np.arctan2(gy[y,x], gx[y,x])/np.pi))+mid
#             magnitude = np.sqrt(gx[y,x]*gx[y,x]+gy[y,x]*gy[y,x])
#             hog[y,x,angle] += magnitude
             
             
#     #build integral image
#     for x in xrange(1, sz[0]-1):
#         for ang in xrange(numorient):
#             hog[y,x,ang] += hog[y,x-1,ang]
#     for y in xrange(1, sz[1]-1):
#         for ang in xrange(numorient):
#             hog[y,x,ang] += hog[y-1,x,ang]
#     for y in xrange(1, sz[1]-1):
#         for x in xrange(1, sz[0]-1):
#             for ang in xrange(numorient):
#                 #tambah kiri dan atas, kurangi dengan kiri-atas
#                 hog[y,x,ang] += hog[y-1,x,ang] + hog[y,x-1,ang] - hog[y-1,x-1,ang]
#     return hog
