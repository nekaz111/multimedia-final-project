# basic program to combine images into video

import cv2 as cv
import numpy as np
import glob

images = []
for filename in glob.glob('Blob_Detection/*.bmp'):
    img = cv.imread(filename)
    h, w, layers = img.shape
    size = (w, h)
    images.append(img)
 
 
video = cv.VideoWriter('footage.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(images)):
    video.write(images[i])
video.release()