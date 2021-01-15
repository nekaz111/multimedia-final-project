#Jeremiah Hsieh Multimedia Final Project Frame Differencing
# footage from https://www.youtube.com/watch?v=_5iVa8koZ7Y

#instead of comparing frame to average background, compare to previous frame and calculate differnce, assume differences == movement
# copied my background subtraction function over except that instead of using background it uses previous frame
import cv2 as cv
import numpy as np






def nothing(x):
    pass


def calculateForeground():
    # subtracting differences in each image and assuming anything different is foreground and anything not different is background
    # cap = cv.VideoCapture('footage.avi')
    cap = cv.VideoCapture('footage.avi')
    
    #find total # of frames in video
    total_frames = cap.get(7)
    # the following code is background subtraction main method
    #after average background is calculated subtract from video playback and assume anything left is foreground
    framecounter = 0
    # subtracted = []
    
    #make trackbars for slider control 
    cv.namedWindow("Trackbars")
    cv.createTrackbar("binary thresh", "Trackbars", 40, 255, nothing)
    cv.createTrackbar("diff thresh", "Trackbars", 10, 120, nothing)
    
    ret, previous = cap.read()
    # threshold for abs difference between forground and background
    
    while(cap.isOpened()):
        ret, current = cap.read()
        #get new values
        low = cv.getTrackbarPos("binary thresh", "Trackbars")
        diffthresh = cv.getTrackbarPos("diff thresh", "Trackbars")
        # loops video
        # if framecounter == total_frames:
        #         # reset back to begenning of video stream
        #         cap.set(1, 0)
        # framecounter += 1

        # just use ret instead since ret returns false if no input? 
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, current = cap.read()
        
        
        hsvP = cv.cvtColor(previous, cv.COLOR_BGR2HSV)
        hsvC = cv.cvtColor(current, cv.COLOR_BGR2HSV)
        
        # # manual method is very inefficient and slow
        # numpied = np.array(current)
        # subtracted = np.array(current)
        # subtracted = np.zeros(current.shape)
        # for x in range(0, current.shape[0]):
        #     for y in range(0, current.shape[1]):
        #         for z in range(0, current.shape[2]):
        #             value = current[x, y, z] - average[x, y, z]
        #             if value < 0:
        #                 subtracted[x, y, z] = 0
        #             elif value > 255:
        #                 subtracted[x, y, z] = 255
        #             else:
        #                 subtracted[x, y, z] = abs(value)
        
    
        
        # calculated difference between foreground and background        
        # absolute difference instead of regular subtraction
        # difference = cv.absdiff(current, average.astype(np.uint8))
        difference = cv.absdiff(current, previous)
        
        ## alternatively just directly set to 0 if absolute difference is less than threshold?
        # current[np.all(abs(current - average.astype(np.uint8) < [diffthresh, diffthresh, diffthresh]), axis = 2)] = 0
        blur = cv.blur(difference,(5,5))
        # immediately threshold?
        colorthresh = cv.inRange(blur, (10, 10, 10), (255, 255, 255))
        # erosion and dilation functions
        # erode first to remove any smaller noise, then dilate back so that any remaining larger blobs are same size
        colorthresh = cv.erode(colorthresh, None, iterations = 2)
        colorthresh = cv.dilate(colorthresh, None, iterations = 2)
        
        
        # difference[difference < diffthresh] = 0
        # difference[np.where(().all(axis = 2))] = [0, 0, 0] 
        # difference[np.all(difference[:, :, 1] < diffthresh and difference[:, :, 2] < diffthresh and difference[:, :, 3] < diffthresh)] = 0
        # all values below threshold are set to 0
        # check all 3 BGR values and they must all be below threshold simultaneously
        
        # difference[np.all(difference < diffthresh, axis = 2)] = 0
        difference[np.all(difference < [diffthresh, diffthresh, diffthresh], axis = 2)] = 0
        

        # grayscale image
        gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
        
        th = 1
        manual =  gray > th
        test = np.zeros_like(previous, np.uint8)
        test[manual] = current[manual]
        
        
        # threshold image to get mask
        ret, thresh = cv.threshold(gray, low, 255, cv.THRESH_BINARY)
        # erosion and dilation functions
        # erode first to remove any smaller noise, then dilate back so that any remaining larger blobs are same size
        thresh = cv.erode(thresh, None, iterations = 2)
        thresh = cv.dilate(thresh, None, iterations = 2)
        # apply mask to original image to get objects
        # current.copyTo(result, ~thresh)
        
        
        # #calculate canny low/high pixel values
        # high, thresh_im = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # low = 0.5 * high
        # # canny edge detection
        # edge = cv.Canny(gray, low, high)

        # # inverted edge image
        # inverted = cv.bitwise_not(edge)
        
        
        # inverted threshold
        # threshinvert = cv.bitwise_not(thresh)
        
        # apply threshold to original image so that all non objects are removed
        final = cv.bitwise_and(current, current, mask = thresh)
        
        
        
        # # draw boxes around ima1ge
        # boxes = connected(current, colorthresh)
        
        
        
        # output results
        cv.imshow('original', current)
        cv.imshow('final',difference)
        # cv.imshow('subtraction', subtracted)
        # cv.imshow('thresholded', thresh)
        # cv.imshow('grayscale', gray)
        # cv.imshow('edge', edge)
        # cv. imshow('inverted edge', inverted)
        cv.imshow('color thresh', colorthresh)
        cv.imshow('testing', final)
        
        cv.imshow('testing', test)
        # cv.imshow('bounding boxes', boxes)
        
        
        
        #exit
        if cv.waitKey(5) & 0xFF == ord('q'):
          break
    
        previous = current
    
    
    
    
    
    # #opencv versoin for comparison
    # fgbg = cv.createBackgroundSubtractorMOG2() 
      
    # while(cap.isOpened()): 
    #     # get frame data
    #     ret, frame = cap.read() 
    #     # make mask 
    #     fgmask = fgbg.apply(frame) 
       
    #     cv.imshow('fgmask', frame) 
    #     cv.imshow('frame', fgmask) 
      
          
    #     if cv.waitKey(5) & 0xFF == ord('q'):
    #         break
  
    cap.release()
    cv.destroyAllWindows()


#draw mask of boxes around individual objects
#basic method using connected component after thresholding
#thresholding is used to convert image to binary equivalent while connected components is used to detected edges and assume continuous edge is an object
#obvious problems - if objects have black on them then threshold will not show edge properly
#any smaller edges within objects would be considered a seperate object since it has its own connected edge
#if the colors of the objects and the background are too similar than thresholding will not find edge properly
def connected(img, thresh):
    connectivity = 4
    components, output, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
    components = components - 1;
    #make empty numpy image to copy
    boxes = np.zeros((img.shape), np.uint8)
    for x in range(0, components + 1):
        #random colors to fill objects  
        color = np.random.randint(255, size = 3)
        #ghetto fix
        #stats returns top left and bottom right x, y coordinates for drawing box
        #remove all boxes udner a certain size
        #or techincally speaking don't add all boxes under a certain size
        topleft = (stats[x][0], stats[x][1])
        botright = (stats[x][0] + stats[x][2], stats[x][1]+stats[x][3])
        if botright[0] - topleft[0] > 50 or botright[1] - topleft[1] > 50:
            #draw bounding box 
            cv.rectangle(boxes, topleft, botright, (0, 0, 255), 2)
        boxes[output == x + 1] = color
    return boxes


def main():
    calculateForeground()
    

if __name__ == "__main__":
    main()