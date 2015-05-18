import cv2
import numpy as np
import sys, time
import imutils
from matplotlib.pyplot import imshow,show
from matplotlib import pyplot
from skimage.feature import hog
from skimage import data, color, exposure

#image is the grayscale image
def detectCircle(image,minRadius=3,maxRadius=100,param2=80,showImage=False):
    # image = cv2.medianBlur(image,5)
    cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # cv2.imshow('1',cimage)
    circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    # print circles.shape

    for i in circles[0,:]:
        cv2.circle(cimage,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
        cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle

    if showImage:
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

def window_and_hog_detection(image,template,(winW,winH)=(80,80),hog_diff_threshold = 0.02):
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


if __name__ == "__main__" :

    # if len(sys.argv)<2:
    #     sample = "roadsign/image5.jpg"
    # else:
    #     sample = "roadsign/image"+sys.argv[1]+".jpg"
    # template_name = "roadsign/template.jpg"

    # if len(sys.argv)<2:
    #     sample = "Starbucks/image11.jpg"
    # else:
    #     sample = "Starbucks/image"+sys.argv[1]+".jpg"
    # template_name = "Starbucks/template.png"

    if len(sys.argv)<2:
        sample = "Superman/image1.jpg"
    else:
        sample = "Superman/image"+sys.argv[1]+".jpg"
    template_name = "Superman/template.png"

    #read as grayscale
    image = cv2.imread(sample,0)
    template = cv2.imread(template_name,0)

    
    # Extract the green channel
    # image[:,:,0] = 0
    # image[:,:,2] = 0
    # template[:,:,0] = 0
    # template[:,:,2] = 0

    # looping(image,(295,216))

    # detectCircle(image,3,100,param2=80,showImage=True)
    # circle_and_hog_detection(image,template)
    # window_and_hog_detection(image,template,(295,216),hog_diff_threshold=0.05)

    # imshow(image,cmap = pyplot.get_cmap('gray'))
    # show()

