#Jeremiah Hsieh Multimedia Final Project Moving Camera
# footage from https://www.youtube.com/watch?v=_5iVa8koZ7Y
# currently just a copy of background subtraction since I did not manage to figure out how to do this without machine learning


import cv2 as cv
import numpy as np


#calculates average background calculation based on video data input over time
def calculateBackgroundRealtime():
    # subtracting differences in each image and assuming anything different is foreground and anything not different is background
    cap = cv.VideoCapture('moving_camera.mp4')
    framecounter = 0
    
    #find total # of frames in video
    total_frames = cap.get(7)
    
    
    # 
    # different alpha values are used to show how it affects the the types of value can impact the accumulated average background value
    # get current frame
    ret, current = cap.read()
    #stores average backgroudn values
    average1 = average2 = average3 = np.float32(current)
    # average2 = np.float32(current)
    
    
    def nothing(x):
        pass
    
    #make trackbars for slider control 
    cv.namedWindow("Trackbars")
    cv.createTrackbar("alpha", "Trackbars", 1, 100, nothing)
    
    
    while(cap.isOpened()):
        # loops video
        framecounter += 1
        if framecounter == total_frames:
                # reset back to begenning of video stream
                cap.set(1, 0)
        alpha = cv.getTrackbarPos("alpha", "Trackbars") / 100
        ret, current = cap.read()
        # calculate accumulated weighted averege where the number (alpha) indicates how long differences in values are kept 
        cv.accumulateWeighted(current, average1, alpha)
        # cv.accumulateWeighted(current, average2, .1)
        # cv.accumulateWeighted(current, average3, .01)
        # convert back to image
        img1 = cv.convertScaleAbs(average1)
        # img2 = cv.convertScaleAbs(average2)
        # img3 = cv.convertScaleAbs(average3)
        cv.imshow('original', current)
        cv.imshow('average1',img1)
        # cv.imshow('average2',img2)
        # cv.imshow('average3',img3)

        #exit
        if cv.waitKey(1) & 0xFF == ord('q'):
          break
    
    
    
    
    # #read image data from cap stream
    # while(cap.isOpened()):
    #   ret, frame = cap.read()
    #   if ret == True:
    #     #show a frame
    #     cv.imshow('capeo',frame)
    #     #exit
    #     if cv.waitKey(25) & 0xFF == ord('q'):
    #       break
    #   else: 
    # #     break
    cap.release()
    cv.destroyAllWindows()
    

# calculates "final" background value arrived at comparison value for video
def calculateBackgroundFinal():
    # subtracting differences in each image and assuming anything different is foreground and anything not different is background
    cap = cv.VideoCapture('moving_camera.mp4')
    
    #find total # of frames in video
    total_frames = cap.get(7)
    #get average values and assume it is background
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    
    
    
    #average stores background average values to compare
    # average = np.empty((w, h, 3), np.dtype('float32'))
    ret, current = cap.read()
    average = np.float32(current)
    #read image data from video stream
    for x in range(1, int(total_frames) , 1):
        #get current frame
        ret, current = cap.read()
        first = current
        
        # # get 2nd frame
        # cap.set(1, x + 1)
        # ret, current = cap.read()
        # second = current
        # subtracted.append(second-first)
        cv.accumulateWeighted(current, average, .01)
        
    img = cv.convertScaleAbs(average)   
    cv.imshow('average', img)
    cv.waitKey(0)
    
    cap.release()
    cv.destroyAllWindows()
    
    return img

# instead of looking at entire video, just look at a random spread of frames or x number of preceding/proceeding frames before/after current frame and assume that background exists in all of them
def calculateBackgroundPartial():
    # subtracting differences in each image and assuming anything different is foreground and anything not different is background
    cap = cv.VideoCapture('moving_camera.mp4')
    #find total # of frames in video
    total_frames = cap.get(7)
    # choose random frames
    randomframes = total_frames * np.random.uniform(size = 30)
     
    
    stored = []
    for random in randomframes:
        # go to random frame
        cap.set(cv.CAP_PROP_POS_FRAMES, random)
        # get image
        ret, current = cap.read()
        # add to array
        stored.append(current)
     
    # calculate background values using random frames
    average = np.median(stored, axis = 0).astype(dtype = np.uint8)    
     
    # show image
    cv.imshow('average', average)
    cv.waitKey(0)
    
    cap.release()
    cv.destroyAllWindows()
    
    return average




def nothing(x):
    pass

# opencv included method to compare to my output
def compareBackground():
    cap = cv.VideoCapture('moving_camera.mp4')
    subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    while(cap.isOpened()):
        ret, current = cap.read()
        mask = subtractor.apply(current)
        cv.imshow("original", current)
        cv.imshow("mask", mask)
        
        
        final = cv.bitwise_and(current, current, mask = mask)
        cv.imshow("final", final)
        #exit
        if cv.waitKey(5) & 0xFF == ord('q'):
          break
      
    cap.release()
    cv.destroyAllWindows()
        

# uses calculated background values and extracts moving objects from video foreground
def calculateForeground(average):
    
    # subtracting differences in each image and assuming anything different is foreground and anything not different is background
    cap = cv.VideoCapture('moving_camera.mp4')
    
    #find total # of frames in video
    total_frames = cap.get(7)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # the following code is background subtraction main method
    #after average background is calculated subtract from video playback and assume anything left is foreground
    framecounter = 0
    # subtracted = []
    
    #make trackbars for slider control 
    cv.namedWindow("Trackbars")
    cv.createTrackbar("binary thresh", "Trackbars", 40, 255, nothing)
    cv.createTrackbar("diff thresh", "Trackbars", 10, 120, nothing)
    
    ret, result = cap.read()
    # threshold for abs difference between forground and background
    
    while(cap.isOpened()):
        #get new values
        low = cv.getTrackbarPos("binary thresh", "Trackbars")
        diffthresh = cv.getTrackbarPos("diff thresh", "Trackbars")
        # loops video
        framecounter += 1
        if framecounter == total_frames:
                # reset back to begenning of video stream
                cap.set(1, 0)
        
        ret, current = cap.read()
        
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
        difference = cv.absdiff(current, average.astype(np.uint8))
        
        ## alternatively just directly set to 0 if absolute difference is less than threshold?
        # current[np.all(abs(current - average.astype(np.uint8) < [diffthresh, diffthresh, diffthresh]), axis = 2)] = 0
        
        # immediately threshold?
        difference[difference < diffthresh] = 0
        # difference[np.where(().all(axis = 2))] = [0, 0, 0] 
        # difference[np.all(difference[:, :, 1] < diffthresh and difference[:, :, 2] < diffthresh and difference[:, :, 3] < diffthresh)] = 0
        # all values below threshold are set to 0
        # check all 3 BGR values and they must all be below threshold simultaneously
        
        # difference[np.all(difference < diffthresh, axis = 2)] = 0
        # difference[np.all(difference < [diffthresh, diffthresh, diffthresh], axis = 2)] = 0
        

        # grayscale image
        gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
        
        
        th = 1
        manual =  gray > th
        test = np.zeros_like(average, np.uint8)
        test[manual] = current[manual]
        
        
        
        # threshold image to get mask
        ret, thresh = cv.threshold(gray, low, 255, cv.THRESH_BINARY)
        # erosion and dilation functions
        # erode first to remove any smaller noise, then dilate back so that any remaining larger blobs are same size
        thresh = cv.erode(thresh, None, iterations = 1)
        thresh = cv.dilate(thresh, None, iterations = 1)
        # apply mask to original image to get objects
        # current.copyTo(result, ~thresh)
        
        # use floodfill to complete blob
        # does not work in this case because blobs are not enclosed shapes
        filled = thresh.copy()
        height, width = thresh.shape[:2]
        floodmask = np.zeros((height + 2, width + 2), np.uint8)
        cv.floodFill(filled, floodmask, (0,0), 255);
        
        
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
        
        
        
        # # draw boxes around image
        # boxes = connected(current, thresh)
        
        
        
        # confirm detected movement
        # check ratio of detected movement (ie. white vs total ratio)
        white = cv.countNonZero(thresh)
        black = h * w - white
        # assume if there is at least 10% motion in frame then something is moving
        if(white/(white + black) > .1):   
            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(current, 'movement detected', (30,30), font, 3, (0, 255, 0), 2, cv.LINE_AA)
        else:
            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(current, 'no movement detected', (30,30), font, 3, (0, 255, 0), 2, cv.LINE_AA)
        
        
        # output results
        cv.imshow('original', current)
        cv.imshow('difference',difference)
        # cv.imshow('subtraction', subtracted)
        cv.imshow('thresholded', thresh)
        # cv.imshow('filled threshold', filled)
        cv.imshow('grayscale', gray)
        # cv.imshow('edge', edge)
        # cv. imshow('inverted edge', inverted)
        cv.imshow('final output', final)
        cv.imshow('final output alt', test)
        # cv.imshow('bounding boxes', boxes)
        #exit
        if cv.waitKey(5) & 0xFF == ord('q'):
          break
    
    
    
    
    
    
    
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







# runs above functions 
def main():
    # calculateBackgroundRealtime()
    # instead of calculating the background over the entire video, calculate  the average based on background 10 seconds before and after?
    average = calculateBackgroundFinal()
    # average = calculateBackgroundPartial()
    # comparative background subtraction function from opencv
    compareBackground()
    calculateForeground(average)
    

if __name__ == "__main__":
    main()