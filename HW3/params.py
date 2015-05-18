******************************************************************
RoadSign
******************************************************************
No.5
detectCircle(image,20,150,param2=120,showImage=True)

No.10
def detectCircle(image,minRadius=50,maxRadius=250,showImage=False):


******************************************************************
Starbucks
******************************************************************
No.3

def detectCircle(image,minRadius=50,maxRadius=100):
    # image = cv2.medianBlur(image,5)
    cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # cv2.imshow('1',cimage)
    circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=120,minRadius=minRadius,maxRadius=maxRadius)


No.4
def detectCircle(image,minRadius=50,maxRadius=250):
    # image = cv2.medianBlur(image,5)
    cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # cv2.imshow('1',cimage)
    circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=100,minRadius=minRadius,maxRadius=maxRadius)

No.5
def detectCircle(image,minRadius=30,maxRadius=150,param2=120,showImage=False):

No.8
detectCircle(image,5,150,param2=120,showImage=True)

No.10
5,250

No.11
def detectCircle(image,minRadius=3,maxRadius=100,param2=80,showImage=False):

No.12
    detectCircle(gray_image,10,100,param2=80,showImage=True)


******************************************************************
Superman
******************************************************************
