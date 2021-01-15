#Jeremiah Hsieh Multimedia Final Project Image/Video Object Detection
#various methods of detecting objects in image
#the following methods only work on obviously dilineated objects currently
#does not work on harder image since the settings for methods basically filter out anything with white
#methods used: thresholding, canny edge detection, 

# this was the portion presented at the halfway point

import cv2 as cv
import numpy as np





#read image
img = cv.imread('ez_detection.png')
#img = cv.imread('shapes.jpg')
# img = cv.imread("harder_detection.jpg")
#invert image
#inverted = cv.bitwise_not(img)
#grayscale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#HSV image (hue, saturation, value)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#split into color channels
h, s, v = cv.split(hsv)

#threshold each channel
ret, threshH = cv.threshold(h,60,255,cv.THRESH_BINARY)
ret, threshS = cv.threshold(s,60,255,cv.THRESH_BINARY)
ret, threshV = cv.threshold(v,60,255,cv.THRESH_BINARY)




#otsu's method (?) to calculate canny low/high pixel values
high, thresh_im = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
low = 0.5 * high

#edge detection with canny edge
edge = cv.Canny(gray, low, high)
inverted = cv.bitwise_not(edge)


#thresholding
#ret, thresh = cv.threshold(gray,60,255,cv.THRESH_BINARY)
ret, thresh = cv.threshold(gray,60,255,cv.THRESH_BINARY)

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
def connected(thresh):
    connectivity = 4
    components, output, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
    # sizes = stats[1:, -1]; 
    components = components - 1;
    #threshhold value for objects in scene
    # min_size = 250 
    #make empty numpy image to copy
    boxes = np.zeros((img.shape), np.uint8)
    for x in range(0, components + 1):
        #random colors to fill objects  
        color = np.random.randint(255, size = 3)
        #ghetto fix
        #stats returns top left and bottom right x, y coordinates for drawing box
        #remove all boxes udner a certain size
        #or techincally speaking don't add all boxes under a certain size
        topleft = (stats[x][0],stats[x][1])
        botright = (stats[x][0]+stats[x][2],stats[x][1]+stats[x][3])
        if botright[0] - topleft[0] > 30 or botright[1] - topleft[1] > 40:
            #draw bounding box 
            cv.rectangle(boxes, topleft, botright, (0, 0, 255), 2)
        boxes[output == x + 1] = color
    return boxes

boxes = connected(thresh)

def nothing(x):
    pass

#make trackbars for slider control 
cv.namedWindow("Trackbars")
cv.createTrackbar("thresh", "Trackbars", 60, 255, nothing)




while(True):
    #get new values
    low = cv.getTrackbarPos("thresh", "Trackbars")
    ret, thresh = cv.threshold(gray, low, 255, cv.THRESH_BINARY)
    boxes = connected(thresh)

    #write image
    cv.imwrite("detected.jpg", boxes) 
    
    #apply mask to original image
    
    #load and implement with video player
    
    #display images
    cv.imshow('objects image',img)
    
    #cv.imshow('inverted image',inverted)
    
    #cv.imshow('gray image',gray)
    #
    #
    cv.imshow('edge image',edge)
    
    cv.imshow('edge inverted image',inverted)
    
    cv.imshow('threshold image',thresh)
    #
    #
    #cv.imshow('outline image', outline)
    #
    cv.imshow('test', boxes)
    
    cv.imshow('hsv image', hsv)
    
    # close windows
    if cv.waitKey(1000) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()