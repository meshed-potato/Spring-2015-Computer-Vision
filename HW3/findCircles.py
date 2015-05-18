import cv2
import numpy as np
import sys

def detectCircle(img,minRadius,maxRadius):
    # img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # cv2.imshow('1',cimg)
    #circles: 1*num*3
    circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=100,minRadius=50,maxRadius=250)
    circles = np.uint16(np.around(circles))
    print circles.shape
    print circles[0][1]

    for i in circles[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circles



if __name__ == "__main__" :
    if len(sys.argv)<2:
        # print "cannot open ",filename
        sample = "Starbucks/image11.jpg"
    else:
        sample = "Starbucks/image"+sys.argv[1]+".jpg"
    img = cv2.imread(sample,0)

    circles = detectCircle(img,50,250)

    




