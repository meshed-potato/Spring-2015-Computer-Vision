# from pylab import plot,show,imread,imshow,figure,subplot
from numpy import vstack,array,reshape,uint8,flipud,arange
from scipy.cluster.vq import kmeans,vq
from matplotlib import colors,pyplot
# import matplotlib.image as mpimg
from matplotlib.pyplot import imshow,imread,figure,subplot,plot,show,bar,hist
import math,sys,numpy,Image,ImageDraw,scipy
import scipy.ndimage as ndi

from scipy.cluster.vq import kmeans,vq

IMG_MAX_VALUE = 255.0
#Write a function that gets as input an RGB image, quantizes the colors in the image via k-means, and then replaces 
#the color at each pixel with the color of its quantized version. The number of clusters k is a variable of your function.

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def test(image_name):
    image = Image.open(image_name).convert("L")
    mag_sup,gradx,grady = edgeDetection(image)
    imshow(mag_sup,cmap = pyplot.get_cmap('gray'))
    show()

def q1(image_name,radius,threshold_value):
    # err = 0.2

    img = imread(image_name)
    gray_img = rgb2gray(img)
    #threshold
    gray_img[gray_img<threshold_value] = 0

    # vote_data = gray_img*0
    vote_data = numpy.zeros(shape=(gray_img.shape[0],gray_img.shape[1]))

    for i in range(gray_img.shape[0]):
        print i
        for j in range(gray_img.shape[1]):
            if(gray_img[i][j]>0):
                for dx in range(-radius,radius+1):
                    dy = math.sqrt(radius**2-dx**2)
                    dy = math.floor(dy+0.5)
                    # if(i-dx>0 and j-dy>0):
                    #     vote_data[i-dx][j-dy]+=1
                    # if(i-dx>0 and j+dy<gray_img.shape[1]):
                    #     vote_data[i-dx][j+dy]+=1
                    if(0<i+dx and i+dx<gray_img.shape[0] and j-dy>0):
                        vote_data[i+dx][j-dy]+=1
                    if(0<i+dx and i+dx<gray_img.shape[0] and j+dy<gray_img.shape[1]):
                        vote_data[i+dx][j+dy]+=1

                # lower_x = max(0,i-radius)
                # upper_x = min(gray_img.shape[0],i+radius)
                # lower_y = max(0,j-radius)
                # upper_y = min(gray_img.shape[1],j+radius)
                # for m in range(lower_x,upper_x):
                #     for n in range(lower_y,upper_y):
                #         d_sum = abs(m-i)+abs(n-j)
                #         if d_sum<radius or d_sum>1.42*radius:
                #             pass
                #         else:
                #             diff = math.sqrt( (m-i)**2 + (n-j)**2) - radius
                #             if abs(diff) < err:
                #                 vote_data[m][n]+=1
    #confidence level 80%
    vote_data[vote_data<radius*2*0.25] = 0

    print vote_data
    image = Image.open(image_name)
    draw = ImageDraw.Draw(image)
    for x in range(vote_data.shape[0]):
        for y in range(vote_data.shape[1]):
            if vote_data[x][y] > 0:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius),outline = '#ff0000')
    image.show()

    # vote_data = vote_data*255/vote_data.max()

    # figure(1)
    # subplot(211)
    # imshow(gray_img,cmap = pyplot.get_cmap('gray'))
    # subplot(212)
    # imshow(vote_data,cmap = pyplot.get_cmap('gray'))
    # show()

#with gradient
def q2(image_name,radius,threshold_value):
    err = 0.2
    gradient_bonus = radius*0.4

    img = imread(image_name)
    gray_img = rgb2gray(img)
    #threshold
    gray_img[gray_img<threshold_value] = 0

    vote_data = gray_img*0

    for i in range(gray_img.shape[0]):
        print i
        for j in range(gray_img.shape[1]):
            if(gray_img[i][j]>0):
                gx = gray_img[min(gray_img.shape[0]-1,i+1)][j]-gray_img[max(0,i-1)][j]
                gy = gray_img[i][max(0,j-1)] - gray_img[i][min(gray_img.shape[1]-1,j+1)]
                for dx in range(radius+1):
                    dy = math.sqrt(radius**2-dx**2)
                    dy = math.floor(dy+0.5)
                    if(i-dx>0 and j-dy>0):
                        vote_data[i-dx][j-dy]+=1
                    if(i-dx>0 and j+dy<gray_img.shape[1]):
                        vote_data[i-dx][j+dy]+=1
                    if(i+dx<gray_img.shape[0] and j-dy>0):
                        vote_data[i+dx][j-dy]+=1
                    if(i+dx<gray_img.shape[0] and j+dy<gray_img.shape[1]):
                        vote_data[i+dx][j+dy]+=1
                    #Bonus for gradient
                    if(dx==0 or gx==0):
                        pass
                        # if(dx==0 and gx==0):
                        #     if gy<0:
                        #         vote_data[i][j+dy]+=gradient_bonus
                        #     else:
                        #         vote_data[i][j-dy]+=gradient_bonus
                    elif(abs(gy*1.0/gx - dy*1.0/dx)<err):
                        if(i-dx>0 and j+dy<gray_img.shape[1]):
                            vote_data[i-dx][j+dy]+=gradient_bonus
                        if(i+dx<gray_img.shape[0] and j-dy>0):
                            vote_data[i+dx][j-dy]+=gradient_bonus
                    elif(abs(gy*1.0/gx + dy*1.0/dx)<err):
                        if(i-dx>0 and j-dy>0):
                            vote_data[i-dx][j-dy]+=gradient_bonus
                        if(i+dx<gray_img.shape[0] and j+dy<gray_img.shape[1]):
                            vote_data[i+dx][j+dy]+=gradient_bonus

    vote_data[vote_data<radius*2*2.5] = 0
    # vote_data = 255-vote_data
    # print vote_data
    # vote_data = vote_data*255/vote_data.max()

    # figure(1)
    # subplot(211)
    # imshow(gray_img,cmap = pyplot.get_cmap('gray'))
    # subplot(212)
    # imshow(vote_data,cmap = pyplot.get_cmap('gray'))
    # show()

    image = Image.open(image_name)
    draw = ImageDraw.Draw(image)
    for x in range(vote_data.shape[0]):
        for y in range(vote_data.shape[1]):
            if vote_data[x][y] > 50:
                draw.ellipse((y-radius, x-radius, y+radius, x+radius),outline = 'red')
    image.show()


#with edge and gradient
def q3(image_name,radius,threshold_value):
    err = 0.2
    gradient_bonus = radius*0.4

    image = Image.open(image_name).convert("L")
    mag_sup,gradx,grady = edgeDetection(image)

    img = imread(image_name)
    # gray_img = rgb2gray(img)
    #threshold
    # gray_img[gray_img<threshold_value] = 0
    #gray_img and edge
    gray_img = mag_sup 

    vote_data = gray_img*0

    for i in range(gray_img.shape[0]):
        print i
        for j in range(gray_img.shape[1]):
            if(gray_img[i][j]>0):
                gx = gray_img[min(gray_img.shape[0]-1,i+1)][j]-gray_img[max(0,i-1)][j]
                gy = gray_img[i][max(0,j-1)] - gray_img[i][min(gray_img.shape[1]-1,j+1)]
                for dx in range(radius+1):
                    dy = math.sqrt(radius**2-dx**2)
                    dy = math.floor(dy+0.5)
                    if(i-dx>0 and j-dy>0):
                        vote_data[i-dx][j-dy]+=1
                    if(i-dx>0 and j+dy<gray_img.shape[1]):
                        vote_data[i-dx][j+dy]+=1
                    if(i+dx<gray_img.shape[0] and j-dy>0):
                        vote_data[i+dx][j-dy]+=1
                    if(i+dx<gray_img.shape[0] and j+dy<gray_img.shape[1]):
                        vote_data[i+dx][j+dy]+=1
                    #Bonus for gradient
                    # if(dx==0 or gx==0):
                    #     pass
                    #     # if(dx==0 and gx==0):
                    #     #     if gy<0:
                    #     #         vote_data[i][j+dy]+=gradient_bonus
                    #     #     else:
                    #     #         vote_data[i][j-dy]+=gradient_bonus
                    # elif(abs(gy*1.0/gx - dy*1.0/dx)<err):
                    #     if(i-dx>0 and j+dy<gray_img.shape[1]):
                    #         vote_data[i-dx][j+dy]+=gradient_bonus
                    #     if(i+dx<gray_img.shape[0] and j-dy>0):
                    #         vote_data[i+dx][j-dy]+=gradient_bonus
                    # elif(abs(gy*1.0/gx + dy*1.0/dx)<err):
                    #     if(i-dx>0 and j-dy>0):
                    #         vote_data[i-dx][j-dy]+=gradient_bonus
                    #     if(i+dx<gray_img.shape[0] and j+dy<gray_img.shape[1]):
                    #         vote_data[i+dx][j+dy]+=gradient_bonus

    vote_data[vote_data<radius*2*0.4] = 0
    # vote_data = 255-vote_data
    # print vote_data
    # vote_data = vote_data*255/vote_data.max()

#K means test
    image = Image.open(image_name)
    draw = ImageDraw.Draw(image)
    res = numpy.array([])
    for i in range(vote_data.shape[0]):
        for j in range(vote_data.shape[1]):
            if vote_data[i][j]>0:
                res = numpy.append(res,[i,j])
    res = reshape(res,(res.shape[0]/2,2))
    k = 13
    print res
    centroids,_ = kmeans(res,k)
    print centroids.shape
    for i in range(k):
        x = centroids[i][0]
        y = centroids[i][1]
        draw.ellipse((y-radius, x-radius, y+radius, x+radius),outline = 'red')
    # image = Image.open(image_name)
    # draw = ImageDraw.Draw(image)
    # for x in range(vote_data.shape[0]):
    #     for y in range(vote_data.shape[1]):
    #         if vote_data[x][y] > 0:
    #             draw.ellipse((y-radius, x-radius, y+radius, x+radius),outline = 'red')
    # # image.show()

    figure(1)
    subplot(311)
    imshow(gray_img,cmap = pyplot.get_cmap('gray'))
    subplot(312)
    imshow(vote_data,cmap = pyplot.get_cmap('gray'))
    subplot(313)
    imshow(numpy.asarray(image))

    show()


def edgeDetection(im):
    sigma = 1.4 
#   imdata = numpy.array(im)
    imdata = numpy.array(im, dtype = float)
    # image: 494 * 343 ---> but Image(x, y) will be represent in array like Arr[y][x], so Arr will become 343 * 494, so Arr[x][y] will refer to the image(x,y)
    # G = imdata
    G = ndi.filters.gaussian_filter(imdata, sigma)
#   G = numpy.transpose(imdata)
#   numpy.transpose(G)
    

    gradOut = Image.new('L', im.size)
    gradx = numpy.array(gradOut, dtype = float)   
    grady = numpy.array(gradOut, dtype = float)   
#   graddxdy = numpy.array(gradOut, dtype = float)   

    kernel_x = [[-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1]] 

    kernel_y = [[-1, -2, -1],
                [ 0,  0,  0], 
                [-1, -2, -1]]

    # numpy.array will change image from 424 * 343 to 343*424 array
    width = im.size[1] 
    height = im.size[0] 
    # print width, height

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
#           graddxdy [x][y] = py * 1.0 / px
    
    sobeloutmag = scipy.hypot(gradx, grady)
    sobeloutmag_n = sobeloutmag / numpy.max(sobeloutmag)
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
    return  mag_sup/numpy.max(mag_sup), gradx, grady  


if __name__ == "__main__":
    image3_name = 'HW1/colorful3.png'
    image2_name = 'HW1/colorful2.jpg'
    image1_name = 'HW1/colorful1.jpg'
    image_MoonCraters = 'HW1/MoonCraters.jpg'
    image_planets = 'HW1/planets.jpg'

    # use_img_name = image_MoonCraters
    use_img_name = image3_name
    
    if sys.argv[1] == "q1":
        q1(use_img_name,30,0.5) 
    elif sys.argv[1] == "q2":
        q2(use_img_name,38,0.5)
    elif sys.argv[1] == "q3":
        q3(use_img_name,42,0.5)
    elif sys.argv[1] == "test":
        test(use_img_name)