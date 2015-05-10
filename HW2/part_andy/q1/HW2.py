import cv2
import numpy as np
import sys
import re
from scipy import *
from scipy.linalg import *
from scipy.special import *
from random import choice
from scipy import linalg

def ransac(points_list, trans_type = 'affine', iters = 200, error = 200, good_model_num = 8):
	'''
		This function uses RANSAC algorithm to estimate the
		shift and rotation between the two given images
	'''
	
	model_error = 255
	model_H = None

	for i in range(iters):
		consensus_set = []
		points_list_temp = copy(points_list).tolist()
		# Randomly select 3 points
		temp = []
		for j in range(3):
			if not temp or temp in consensus_set:
				temp = choice(points_list_temp)
			consensus_set.append(temp)
			points_list_temp.remove(temp)
		
		# Calculate the homography matrix from the 3 points
		
		fp0 = []
		fp1 = []
		fp2 = []
		
		tp0 = []
		tp1 = []
		tp2 = []
		for line in consensus_set:
		
			fp0.append(line[0][0])
			fp1.append(line[0][1])
			fp2.append(1)
			
			tp0.append(line[1][0])
			tp1.append(line[1][1])
			tp2.append(1)
			
		fp = array([fp0, fp1, fp2])
		tp = array([tp0, tp1, tp2])

		H = Haffine_from_points(fp, tp ,trans_type)

		# Check if the other points fit this model
		for p in points_list_temp:
			x1, y1 = p[0]
			x2, y2 = p[1]

			A = array([x1, y1, 1]).reshape(3,1)
			B = array([x2, y2, 1]).reshape(3,1)
			
			out = B - dot(H, A)
			dist_err = hypot(out[0][0], out[1][0])
			if dist_err < error:
				consensus_set.append(p)			
		
		min_error_sum = sys.maxint
		# Check how well is our speculated model
		if len(consensus_set) >= good_model_num:
			dists = []
			for p in consensus_set:
				x0, y0 = p[0]
				x1, y1 = p[1]
				
				A = array([x0, y0, 1]).reshape(3,1)
				B = array([x1, y1, 1]).reshape(3,1)
				
				out = B - dot(H, A)
				dist_err = hypot(out[0][0], out[1][0])
				dists.append(dist_err)

			if (max(dists) < error) and (max(dists) < model_error):
				model_error = max(dists)
				model_H = H

#			this_error = max(dists)
#			if this_error <= min_error_sum:
#				print 'this_error\n',this_error
#				min_error_sum = this_error
#				model_H = H

	print 'ransac consensus_set'
	print consensus_set	
	print 'model_H,\n',model_H
	return model_H



def Haffine_from_points(fp, tp, trans_type = 'affine'):
	""" find H, affine transformation, such that 
		tp is affine transf of fp"""

	if fp.shape != tp.shape:
		raise RuntimeError, 'number of points do not match'

	#condition points
	#-from points-
	m = mean(fp[:2], axis=1)
	maxstd = max(std(fp[:2], axis=1))+1e-9
	C1 = diag([1/maxstd, 1/maxstd, 1]) 
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp_cond = dot(C1,fp)

	#-to points-
	m = mean(tp[:2], axis=1)
	C2 = C1.copy() #must use same scaling for both point sets
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp_cond = dot(C2,tp)

	if (trans_type == 'affine'):
	
		#conditioned points have mean zero, so translation is zero
		A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
		U,S,V = linalg.svd(A.T)
	
		#create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
		tmp = V[:2].T
		B = tmp[:2]
		C = tmp[2:4]
	
		tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1) 
		H = vstack((tmp2,[0,0,1]))
	
		#decondition
		H = dot(linalg.inv(C2),dot(H,C1))

	elif trans_type == 'homography':
		# create matrix for linear method, 2 rows for each correspondence pair
	    nbr_correspondences = fp.shape[1]
	    A = zeros((2*nbr_correspondences,9))
	    for i in range(nbr_correspondences):        
	        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
	                    tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
	        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
	                    tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
	    
	    U,S,V = linalg.svd(A)
	    H = V[8].reshape((3,3))    
	    
	    # decondition
	    H = dot(linalg.inv(C2),dot(H,C1))
	    
	    # normalize and return
	
	return H / H[2][2]

def findInterestPoints(filename):
	keynames = re.split(r'\.', filename)
	
	img = cv2.imread(filename)
	gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(gray, None)
	
	img = cv2.drawKeypoints(gray,kp)
	cv2.imwrite(keynames[0] + "_keypts.jpg", img)

	return kp, des

def findMatches(filename1, filename2):

	kp1, des1 = findInterestPoints(filename1)	
	kp2, des2 = findInterestPoints(filename2)

	kp_matches1 = []
	kp_matches2 = []
	kp_matches = []
	des_matches = []
	
	for i in range(0, des1.shape[0]):
		best_match = sys.maxint 
		bm_index = -1	
		second_match = sys.maxint
		sm_index = -1
		for j in range(0, des2.shape[0]):
			d = sum((des1[i] - des2[j])**2)
			if (d < best_match):
				second_match = best_match
				sm_index = bm_index
				best_match = d
				bm_index = j 
			elif (d < second_match):
				second_match = d
				sm_index = j
		if (best_match / second_match) < 0.8:
			kp_matches1.append(kp1[i].pt)
			kp_matches2.append(kp2[bm_index].pt)
			kp_matches.append([kp1[i].pt, kp2[bm_index].pt])

	# print kp_matches
	
	return kp_matches1, kp_matches2, kp_matches, des_matches

def showMatches(matches1, matches2, image1, image2):
	rows1 = image1.shape[0]		
	rows2 = image2.shape[0]
	col1 = image1.shape[1]
	
	if rows1 < rows2:
		image1 = np.concatenate((image1, np.zeros((rows2 - rows1, image1.shape[1],3))), axis = 0)
	else:
		image2 = np.concatenate((image2, np.zeros((rows1 - rows2, image2.shape[1],3))), axis = 0)

	res = np.concatenate((image1, image2), axis = 1)
	print len(matches1),len(matches2)
	for i in range(len(matches1)):
		cv2.line(res,( int(matches1[i][0]),int(matches1[i][1])),(col1+int(matches2[i][0]),int(matches2[i][1])),(0,255,0),1)
	
	return res

def affineMatches(matches, image1, image2, trans_type) :
	# print matches
	H = ransac(matches, trans_type)
	matches_affine = []
	matches_affine1 = []
	matches_affine2 = []
	err = 0

	for mat in matches:
		vector1 = list(mat[0])
		vector2 = list(mat[1])
		vector1.append(1)
		vector2_est = np.dot(H, vector1)

		if (((1-vector2_est[0]/vector2[0]) ** 2 + (1-vector2_est[1]/vector2[1]) ** 2 ) < 0.2):
		# if abs(vector2_est[0]-vector2[0]) + abs(vector2_est[1]-vector2[1]) < 50:
			matches_affine1.append((vector1[0], vector1[1]))
			matches_affine2.append((vector2[0], vector2[1]))
			matches_affine.append([(vector1[0], vector1[1]), (vector2[0], vector2[1])]) 
			err += sqrt((vector2_est[0] - vector2[0])**2 + (vector2_est[1] - vector2[1])**2)

	result_image = showMatches(matches_affine1, matches_affine2, image1, image2)	

	return result_image, H, err / len(matches) 

def alignImages(img1,img2,transformation,image_name):
	merged_img = np.zeros(img2.shape)
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			original = [i,j,1]
			projected = np.dot(transformation, original)
			if (projected[0]<=img2.shape[0] and projected[1]<=img2.shape[1]):
				merged_img[int(projected[0])][int(projected[1])] = img1[i][j]
				merged_img[int(projected[0])][int(projected[1])][0] = 0
	cv2.imwrite('warped_img.jpg',merged_img)
	for i in range(img2.shape[0]):
		for j in range(img2.shape[1]):
			merged_img[i][j][0] = img2[i][j][0]
	cv2.imwrite(image_name,merged_img)


if __name__ == "__main__":

	img1 = cv2.imread('StopSign1.jpg')
	img2 = cv2.imread('StopSign2.jpg')

	kps1, kps2, kps, des_matches = findMatches('StopSign1.jpg', 'StopSign2.jpg')

	result_img12 = showMatches(kps1, kps2, img1, img2)
	cv2.imwrite('result13.png', result_img12)
	
	result_img12_affine, H_affine, avgError = affineMatches(kps, img1, img2, 'affine')
	cv2.imwrite('result14_affine.png', result_img12_affine)
	alignImages(img1,img2,H_affine,'merged_img_affine.jpg')

#	# Q7	
#	print "Average Error" + str(avgError)

	result_img12_Homography, H_homo, avgError = affineMatches(kps, img1, img2, 'homography')
	cv2.imwrite('result13_homography.png', result_img12_Homography)
	alignImages(img1,img2,H_homo,'merged_img_homo.jpg')
	
	# Q7	
	print "Average Error" + str(avgError)
