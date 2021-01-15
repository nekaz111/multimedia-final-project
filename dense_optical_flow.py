#example tutorial program taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
#dense optical flow
# not considered part of the paper since I was still trying to figure out ho wthis works

import numpy as np
import cv2 as cv


cap = cv.VideoCapture("videoplayback.mp4")

ret, frame1 = cap.read()
# greyscale of first frame
gray1 = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(cap.isOpened()):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(gray1,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    cv.imshow('frame2',rgb)

    if cv.waitKey(5) & 0xFF == ord('q'):
      break
    prvs = next

cap.release()
cv.destroyAllWindows()