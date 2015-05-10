# from pylab import plot,show,imread,imshow,figure,subplot
from numpy import vstack,array,reshape,uint8,flipud,arange
from scipy.cluster.vq import kmeans,vq
from matplotlib import colors,pyplot
# import matplotlib.image as mpimg
from matplotlib.pyplot import imshow,imread,figure,subplot,plot,show,bar,hist,title
import math,sys,numpy

IMG_MAX_VALUE = 255.0
#Write a function that gets as input an RGB image, quantizes the colors in the image via k-means, and then replaces 
#the color at each pixel with the color of its quantized version. The number of clusters k is a variable of your function.
def q1(img,k):
    # data generation
    data = reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))

    # performing the clustering
    centroids,_ = kmeans(data,k) # k colors will be found
    # quantization
    qnt,_ = vq(data,centroids)

    # reshaping the result of the quantization
    centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
    clustered = centroids[centers_idx]

    return clustered


#Write a function that gets an RGB image as input and converts it to HSV color space. 
#The function then quantizes only the H channel and replaces every pixel in it with the value of its quantized version. 
#Next, take the quantized H channel and the original S,V channels and convert the image back into RGB.
def q2(img,k):
    # data = reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
    img_hsv = colors.rgb_to_hsv(img*1.0/IMG_MAX_VALUE)
    H = img_hsv[:,:,0]
    H_data = reshape(H,(H.shape[0]*H.shape[1],1))

    centroids,_ = kmeans(H_data,k) # six colors will be found
    qnt,_ = vq(H_data,centroids)

    # reshaping the result of the quantization
    centers_idx = reshape(qnt,(H.shape[0],H.shape[1]))
    clustered = centroids[centers_idx]

    quanted_hsv = img_hsv
    quanted_hsv[:,:,0] = clustered[:,:,0]
    quanted_img = colors.hsv_to_rgb(quanted_hsv)*IMG_MAX_VALUE

    return quanted_img.astype(uint8)

#Write a function that computes the SSD (Sum of Squared Distances) between two RGB images.
def q3(img1,img2):
    diff = img1 - img2
    diff2 = diff*diff
    ssd = diff2.sum()
    return ssd

#Write a function that gets as input an RGB image and draws the histogram of the H channel in two ways. First, 
#when the histogram bins are spaced regularly. Second, when the bins are
#determined according to the colors you get when quantizing the H channel.
def q4(img,k):
    # data = reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
    img_hsv = colors.rgb_to_hsv(img*1.0/IMG_MAX_VALUE)
    H = img_hsv[:,:,0]
    H_data = reshape(H,(H.shape[0]*H.shape[1],1))

    H_centroids,_ = kmeans(H_data,k) # k colors will be found
    H_qnt,_ = vq(H_data,H_centroids)
    H_centers_idx = reshape(H_qnt,(H.shape[0],H.shape[1]))
    H_clustered = H_centroids[H_centers_idx]

    return H_data,H_clustered,H_centroids

#Write a function that calls all the functions above and presents the histograms of the images
#before and after quantization. The function should display the images and report the SSD between 
#the original image and its quantized version. It should do this for two values of k (you can choose the values). 
#Please make sure each result has a clear title and explanation.
def q5(image_name,k):
    img = imread(image_name)
    if(img.shape[2]==4 and image_name.endswith(".png")):
        img = numpy.delete(img,(3),axis=2)*IMG_MAX_VALUE
    img_clustered = q1(img,k)

    ssd = q3(img_clustered,img)
    print ssd

    H_data, H_clustered, H_centroids = q4(img,k)
    figure(1)

    subplot(321)
    title('RGB quantization(k='+str(k)+')[Original Image]')
    imshow(img.astype(uint8))
    subplot(322)
    title('SSD: '+str(ssd)+'[Quantized Image]')
    imshow(img_clustered.astype(uint8))
    
    quanted_img = q2(img,k)
    ssd2 = q3(quanted_img,img)

    subplot(323)
    title('H quantization(k='+str(k)+')[Original Image]')
    imshow(img.astype(uint8))
    subplot(324)
    title('SSD : '+str(ssd2)+'[Quantized Image]')
    imshow(quanted_img.astype(uint8))

    subplot(325)
    title('Histogram of H Channel spaced regularly')
    hist(H_data,bins=arange(0,1,.01))

    subplot(326)
    title('Histogram of quantized H Channel')
    umin = max(H_centroids.min()-.1,0)
    umax = min(H_centroids.max()+.1,1)
    hist(H_clustered.flatten(),bins=arange(umin,umax,(umax-umin)*1.0/100)) 
    show()

#For test use
def test(image_name):
    img = imread(image_name)
    if(img.shape[2]==4 and image_name.endswith(".png")):
        img = numpy.delete(img,(3),axis=2)*IMG_MAX_VALUE
    print img.shape
    img_hsv = colors.rgb_to_hsv(img*1.0/IMG_MAX_VALUE)
    img2 = colors.hsv_to_rgb(img_hsv)*IMG_MAX_VALUE
    figure(1)
    subplot(221)
    imshow(img)
    subplot(222)
    imshow(img_hsv)
    subplot(223)
    imshow(img2.astype(uint8))
    show()

if __name__ == "__main__":
    image3_name = 'HW1/colorful3.png'
    image2_name = 'HW1/colorful2.jpg'
    image1_name = 'HW1/colorful1.jpg'

    use_img_name = image3_name
    img = imread(use_img_name)
    if(img.shape[2]==4 and use_img_name.endswith(".png")):
        img = numpy.delete(img,(3),axis=2)*IMG_MAX_VALUE
    use_img_name = sys.argv[1]
    q5(use_img_name,int(sys.argv[2]))
