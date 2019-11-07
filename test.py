#Jeremiah Hsieh Multimedia Final Project Image/Video Object Detection
#various methods of detecting objects in image
#the following methods only work on obviously dilineated objects currently
#does not work on harder image since the settings for methods basically filter out anything with white

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt




#read image
#img = cv.imread('ez_detection.png')
img = cv.imread("harder_detection.jpg")
##invert image
#inverted = cv.bitwise_not(img)
#grayscale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#otsu's method (?) to calculate canny low/high pixel values
high, thresh_im = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
low = 0.5 * high

#edge detection with canny edge
edge = cv.Canny(gray, low, high)
inverted = cv.bitwise_not(edge)


#thresholding
#ret, thresh = cv.threshold(gray,60,255,cv.THRESH_BINARY)
ret, thresh = cv.threshold(gray,80,255,cv.THRESH_BINARY)

#alternate method obtained from internet for comparitive measure against previously used methods
kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion
erosion = cv.erode(thresh, kernel,iterations = 1) #refines all edges in the binary image
opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel)
outline = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image




#draw mask of boxes around individual objects
#basic method using connected component after thresholding
#thresholding is used to convert image to binary equivalent while connected components is used to detected edges and assume continuous edge is an object
#obvious problems - if objects have black on them then threshold will not show edge properly
#any smaller edges within objects would be considered a seperate object since it has its own connected edge
#if the colors of the objects and the background are too similar than thresholding will not find edge properly

connectivity = 4
components, output, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
sizes = stats[1:, -1]; components = components - 1
#threshhold value for objects in scene
min_size = 250 
#make empty numpy image to copy
boxed = np.zeros((img.shape), np.uint8)
for x in range(0, components+1):
    #use if sizes[i] >= min_size: to identify your objects
    color = np.random.randint(255,size = 3)
    #stats returns top left and bottom right x, y coordinates for drawing box
    #remove all boxes udner a certain size
    #or techincally speaking don't add all boxes under a certain size
    topleft = (stats[x][0],stats[x][1])
    botright = (stats[x][0]+stats[x][2],stats[x][1]+stats[x][3])
    if botright[0] - topleft[0] > 30 or botright[1] - topleft[1] > 30:
        #draw bounding box 
        cv.rectangle(boxed, topleft, botright, (0, 0, 255), 2)
    boxed[output == x + 1] = color

#write image
cv.imwrite("detected.jpg", boxed) 

#apply mask to original image

#load and implement with video player

#display images
cv.imshow('objects image',img)

#cv.imshow('inverted image',inverted)

#cv.imshow('gray image',gray)
#
#
cv.imshow('edge image',edge)

#cv.imshow('threshold image',thresh)
#
#
#cv.imshow('outline image', outline)
#
cv.imshow('test', boxed)

cv.waitKey(0)