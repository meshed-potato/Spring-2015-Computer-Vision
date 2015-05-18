import cv2
import numpy as np
import sys, time
import imutils
from matplotlib.pyplot import imshow,show
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure, measure
"""Some Example Methods Usage
***********************************

####CIRCLE and sift EXAMPLE  
cimage = circle_and_sift_detection(gray_image, gray_template, image)
iw_filename = "./res/"+sample[:-4]+"_cs"+sample[-4:]
cv2.imwrite(iw_filename,cimage)
***********************************

####Detect Circle Example
detectCircle(gray_image,10,60,param2=50,showImage=True)
***********************************

####window LOOPING image EXAMPLE 
looping(gray_image,(20,20))
***********************************

####CIRCLE and HOG detection EXAMPLE
circle_and_hog_detection(gray_image,gray_template)
***********************************

####WINDOW and HOG detection EXAMPLE
window_and_hog_detection(gray_image,gray_template,hog_diff_threshold=0.05)

***********************************

####WINDOW and sift EXAMPLE
window_and_sift_detection(red_channel_gray_image, red_channel_gray_template)

***********************************

###compare contours EXAMPLE
compare_contours(image_red_channel,template_red_channel)

***********************************

####template matching EXAMPLE
template = imutils.resize(template, width=215)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_matching(gray_image,gray_template,image)

***********************************

####multi-scale template mathcing EXAMPLE
# gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template_matching(gray_image,gray_template,image)
multi_scale_template_matching(template,gray_template,image,gray_image)


####window_and_template_matching EXAMPLE
# window_and_template_matching(gray_image,gray_template)
"""
(weight_B,weight_G, weight_R ,threshold_B, threshold_G, threshold_R) = (0, 0, 1, 0, 0, 150)
image_red_channel
template_red_channel
red_channel_gray_image
red_channel_template
red_channel_gray_template

#image is the grayscale image
def detectCircle(image,minRadius=50,maxRadius=150,param2=100,showImage=False):
    # image = cv2.medianBlur(image,5)
    cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # cv2.imshow('1',cimage)
    circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    # print circles.shape

    if not showImage:
        return circles
    for i in circles[0,:]:
        cv2.circle(cimage,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
        cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle

    cv2.imshow('detected circles',cimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circles

def cal_hog(image,orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True):
    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
    #                 cells_per_block=(1, 1), visualise=True)
    fd, hog_image = hog(image, orientations, pixels_per_cell,
                    cells_per_block, visualise)
    return fd,hog_image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
 
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
 
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
        # yield the next image in the pyramid
        yield image   

def looping(image,(winW,winH)=(128,128)):
    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
     
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
     
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)  

            # cv2.imshow("Window", window)
            # cv2.waitKey(1)
            # time.sleep(0.025)

def window_and_hog_detection(image,template,(winW,winH)=(128,128),hog_diff_threshold = 0.02):
    window_selected = []
    for resized in pyramid(image, scale=1.2):
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            
            # cv2.imshow("Window", window)
            # cv2.waitKey(1)
            # time.sleep(0.025)
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)  

            if hog_comparision(window,template,hog_diff_threshold):
                print 'finded'
                window_selected.append([x,y])
        clone = resized.copy()
        for (x,y) in window_selected:
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def window_and_sift_detection(image,template,scale=1.3,minSize=(30,30)):
    window_selected = []

    winH = int(min(image.shape[0], image.shape[1]*11.0/15.0) )   #init_winH
    winW = int(winH*15.0/11.0)    #init_winW
    while winW > minSize[1] and winH > minSize[0]:
        window_selected = []    #reset window_selected

        for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
            print 'window.shape\n',window.shape
            print 'H:',winH,'W:',winW
            if window.shape[0] != winH or window.shape[1] != winW:
                print "aaaaaa"
                continue
            
            # cv2.imshow("Window", window)
            # cv2.waitKey(1)
            # print "aaaaaa"
            # time.sleep(0.025)
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)  

            good,found,M = sift(template,window)
            if found:
                print 'finded'
                window_selected.append([x,y,winW,winH,M])

        winW = int(winW/scale)
        winH = int(winH/scale)

        clone = image.copy()

        h,w = template.shape

        for (x,y,width,height,M) in window_selected:
        #     cv2.rectangle(clone, (x, y), (x + width, y + height), (0, 255, 0), 2)
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            dst[:,:,0] += x
            dst[:,:,1] += y
            print dst.shape
            for pt in dst:
                p = pt[0]
                p = list(p)
                cv2.circle(clone, tuple(p), 5, (255, 255, 255), -1)
            cv2.polylines(clone,[np.int32(dst)],True,255,3)

        cv2.imshow("Window", clone)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def circle_and_sift_detection(image,template,original_image):
    circles = detectCircle(image)
    num_of_circles = circles.shape[1]
    circles_selected = []
    for i in range(num_of_circles):
        [c_x,c_y,c_r] = circles[0][i]
        c_r = int(c_r)
        candidate = image[max(0,c_y - c_r):min(image.shape[0],c_y + c_r), max(0,c_x - c_r):min(image.shape[1],c_x + c_r)] 
        print 'radius:\n',c_r
        print'x:',c_x,' y:',c_y
        print '#ofC:',num_of_circles
        # print candidate.shape
        good,found,M = sift(template,candidate)
        if found:
            circles_selected.append([c_x,c_y,c_r,M])

    cimage = original_image

    h,w = template.shape

    for (x,y,r,M) in circles_selected:
        # cv2.circle(cimage,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
        # cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        dst[:,:,0] += x - r
        dst[:,:,1] += y - r
        for pt in dst:
            p = pt[0]
            p = list(p)
            cv2.circle(cimage, tuple(p), 5, (255, 255, 255), -1)

        cv2.polylines(cimage,[np.int32(dst)],True,(255,0,0),3)
        # cv2.imshow('detected',cimage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    cv2_imshow(cimage,'detected circles')
    return cimage


def circle_and_hog_detection(image,template,hog_diff_threshold = 0.05):
    #TODO: Customize winsize according to radius of circle detected

    #Extract subimage accroding to circle detection
    circles = detectCircle(image)
    num_of_circles = circles.shape[1]
    circles_selected = []

    for i in range(num_of_circles):
        # if i >= 1:
        #     break
        [c_x,c_y,c_r] = circles[0][i]
        c_r = int(c_r) #avoid the case of +1 - +2 = 65535
        # candidate = image[c_y - c_r:c_y + c_r, c_x - c_r :c_x + c_r] 
        candidate = image[max(0,c_y - c_r):min(image.shape[1],c_y + c_r), max(0,c_x - c_r):min(image.shape[0],c_x + c_r)] 
        print 'raidus:\n',c_r
        print candidate.shape

        # using hog to compare candidate to template
        if hog_comparision(candidate,template,hog_diff_threshold):
            circles_selected.append([c_x,c_y,c_r])

    cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for i in circles_selected:
        cv2.circle(cimage,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
        cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle
    cv2.imshow('detected circles',cimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#return true if candidate and template are alike
def hog_comparision(candidate,template,hog_diff_threshold):
    fd_candidate,hog_image_candidate = cal_hog(candidate)
    # imshow(hog_image_candidate)
    # imshow(candidate)
    # show()
    #compare with the template hog
    template_resized = imutils.resize(template, width=candidate.shape[0])
    fd_template,hog_image_template = cal_hog(template_resized)

    diff = hog_diff(fd_candidate,fd_template)
    ave_diff = diff/candidate.shape[0]
    print "ave_diff:",ave_diff
    if (ave_diff < hog_diff_threshold):
        return True
    return False

def hog_diff(fd1,fd2):
    len1 = fd1.shape[0]
    len2 = fd2.shape[0]
    if (len1 > len2):
        fd1 = fd1[:len2]
    else:
        fd2 = fd2[:len1]
    return sum( (fd2-fd1) ** 2)

def RGB(im, wB, wG, wR, thB, thG, thR):
    # print im.shape
    h, w, c = im.shape
    b, g, r = cv2.split(im)
    ret, b = cv2.threshold(b, thB, 255, cv2.THRESH_TOZERO)
    ret, g = cv2.threshold(g, thG, 255, cv2.THRESH_TOZERO)
    ret, r = cv2.threshold(r, thR, 255, cv2.THRESH_TOZERO)
    im = cv2.merge((wB * b, wG * g, wR * r))        
    return im, b, g, r 

def color_preProcess(image,weight_B,weight_G, weight_R,threshold_B, threshold_G, threshold_R):
    res, b, g, r = RGB(image, weight_B, weight_G , weight_R, threshold_B, threshold_G, threshold_R)
    return res

#img1 is the template
def sift(img1, img2):
    MIN_MATCH_COUNT = 10
    print img2.shape

    # Initiate sift detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with sift
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

#   img2 = cv2.drawKeypoints(img2,kp2)
#   cv2.imwrite('../results/img2_sift_keypoints.jpg', img2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    found = False
    M = None

    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print 'len(good):\t' + str(len(good))

    if len(good) > MIN_MATCH_COUNT:
        found = True
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # print 'len(src_pts)' + str(len(src_pts))    
        # print 'len(dst_pts)' + str(len(dst_pts))    

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # for pt in dst:
        #     p = pt[0]
        #     p = list(p)
        #     # p = [int(i) for i in p]
        #     print p
        #     cv2.circle(img2, tuple(p), 5, (255, 255, 255), -1)

        # cv2.polylines(img2,[np.int32(dst)],True,255,3)
        # cv2.imshow('detected',img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        found = False
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)

    return good, found, M

def template_matching(img2,template,img):
    w, h = template.shape[::-1]
    print 'img_size',img2.shape
    print 'template_size',template.shape

    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF']


    for meth in methods:
        # img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img2,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)

        # plt.subplot(121),plt.imshow(res,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(img,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)

        # plt.show()
        cv2_imshow(img)
        cv2.imwrite('./res/temp.jpg',img)
    return top_left,bottom_right,w,h

def window_and_template_matching(image,template,scale=1.2,minSize=(30,30)):
    window_selected = []
    for resized in pyramid(image, scale=1.2):
        template_mathcing(resized,template)

def get_contours_after_canny(image):
    
    image_edges = cv2.Canny(image,100,200)

    # plt.subplot(121),plt.imshow(image,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(image_edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    ret,thresh = cv2.threshold(image_edges,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contour_img = np.zeros(image.shape)
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 3)
 

    return contours,contour_img

def cv2_imshow(img,name='NoName'):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_contour(img,cnt,name='NoName'):
    cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    # cv2_imshow(img,name)

def compare_contours(image_red_channel,template_red_channel):
    contours_image, contour_image_img = get_contours_after_canny(image_red_channel)
    contours_template, contour_template_img = get_contours_after_canny(template_red_channel)

    # template_contour_set = [contours_template[51], contours_template[72], contours_template[82], contours_template[125],
                            # contours_template[133],contours_template[155] ]
    template_contour_set = [contours_template[20],contours_template[32],contours_template[40],
                            contours_template[51],contours_template[62],contours_template[116],
                            contours_template[118],contours_template[123],contours_template[136],
                            contours_template[151],contours_template[160]]
    res_contour = []
    for i in range(len(contours_image)):
        cnt1 = contours_image[i]
        if cnt1.shape[0] < 150:
            continue
        print '*****\n'
        for cnt2 in template_contour_set:
            ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
            if ret < 1:
                print ret
                res_contour.append(cnt1)
                temp = image.copy()
                # draw_contour(temp,cnt1,'temp')
    image_clone = image.copy()

    for i in res_contour:
        draw_contour(image_clone,i)
    cv2.imwrite('res/contour.jpg',image_clone)
    # cv2.drawContours(image, res_contour, 3, (0,255,0), 3)
    # cv2_imshow(image_clone)

def multi_scale_template_matching(template,gray_template,image,gray_image):
    template = imutils.resize(template, width=600)
    for resized in pyramid(gray_template, scale=1.2):
        print 'width:',resized.shape[1]
        image_1 = image.copy()
        top_left,bottom_right,w,h = template_matching(gray_image,resized,image_1)
        # extracted_img = image_1[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w] 
        
#Write a function that computes the SSD (Sum of Squared Distances) between two RGB images.
def ssd(img1,img2):
    diff = img1 - img2
    diff2 = diff*diff
    ssd = diff2.sum()
    return ssd

if __name__ == "__main__" :

    inputimage = "Starbucks/image4.jpg"
    template_name = "Starbucks/template.png"
    
    #read as grayscale
    image = cv2.imread(inputimage)
    template = cv2.imread(template_name)


    #resize template
    template = imutils.resize(template, width=image.shape[1])
    ###### Convert to GrayScale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #Generate all images for later usage
    (weight_B,weight_G, weight_R ,threshold_B, threshold_G, threshold_R) = (0, 0, 1, 0, 0, 150)
    image_red_channel = color_preProcess(image, weight_B, weight_G, weight_R, threshold_B, threshold_G, threshold_R)
    template_red_channel = color_preProcess(template,threshold_B, threshold_G, threshold_R, weight_B,weight_G, weight_R)
    red_channel_gray_image = cv2.cvtColor(image_red_channel, cv2.COLOR_BGR2GRAY)
    red_channel_template = color_preProcess(template, weight_B, weight_G, weight_R, threshold_B, threshold_G, threshold_R)
    red_channel_gray_template = cv2.cvtColor(red_channel_template, cv2.COLOR_BGR2GRAY)


    """write the method you wanna try here"""
    #Example
    circle_and_sift_detection(gray_image, gray_template, image)

    """ Some Example Methods Usage
    ***********************************

    ####CIRCLE and sift EXAMPLE  
    cimage = circle_and_sift_detection(gray_image, gray_template, image)
    
    ***********************************

    ####Detect Circle Example
    detectCircle(gray_image,10,60,param2=50,showImage=True)
    ***********************************

    ####window LOOPING image EXAMPLE 
    looping(gray_image,(20,20))
    ***********************************

    ####CIRCLE and HOG detection EXAMPLE
    circle_and_hog_detection(gray_image,gray_template)
    ***********************************

    ####WINDOW and HOG detection EXAMPLE
    window_and_hog_detection(gray_image,gray_template,hog_diff_threshold=0.05)
    ####WINDOW and sift EXAMPLE
    window_and_sift_detection(red_channel_gray_image, red_channel_gray_template)

    ***********************************

    ###compare contours EXAMPLE
    compare_contours(image_red_channel,template_red_channel)

    ***********************************

    ####template matching EXAMPLE
    template = imutils.resize(template, width=215)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_matching(gray_image,gray_template,image)

    ***********************************

    ####multi-scale template mathcing EXAMPLE
    # gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # template_matching(gray_image,gray_template,image)
    multi_scale_template_matching(template,gray_template,image,gray_image)


    ####window_and_template_matching EXAMPLE
    # window_and_template_matching(gray_image,gray_template)
    """


    # contours_image, contour_image_img = get_contours_after_canny(image_red_channel)
    # contours_template, contour_template_img = get_contours_after_canny(template_red_channel)

    # cv2_imshow(contour_image_img)
    # cv2_imshow(contour_template_img)
    # for i in range(len(contours_template)):
    #     cnt2 = contours_template[i]
    #     if cnt2.shape[0] < 200:
    #         continue
    #     print i

    #     template_clone = template.copy()

    #     draw_contour(template_clone,cnt2)

    # print ret
