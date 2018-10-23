import numpy as np
import cv2

#reading data
#video_file = '/home/aayush/Downloads/Vision/Video/Dataset/Fire_test.mp4'
#Test_file neg samples
#video_file = '/home/aayush/Downloads/Vision/Video/Fire_detection/fire_videos.1406/neg/negsVideo10.1072.avi'
#test file pos samples
video_file = '/home/aayush/Downloads/Vision/Video/Fire_detection/fire_videos.1406/pos/posVideo12.879.avi'
#video_file = '/home/aayush/Downloads/Vision/Video/Dataset/fire_videos.1406/pos/Test_fire.mpg'
video = cv2.VideoCapture(video_file)

#processing video frames
count=0
Threshold = 10
while True:
    (ok , frame) = video.read()
    if not ok:
        break
    blur = cv2.GaussianBlur(frame , (21,21) ,0)
    hsv = cv2.cvtColor(blur , cv2.COLOR_BGR2HSV)

    lower = [18 , 50, 50]
    upper = [35 , 255 ,255]

    #convert into array
    lower = np.array(lower , dtype = "uint8")
    upper = np.array(upper , dtype = "uint8")

    mask = cv2.inRange(hsv , lower , upper)
    # creating a bounding box
    frame_copy = frame
    frame_gray = cv2.cvtColor(frame_copy , cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(frame_gray,127,255,0)
    cv2.imshow('Threshold', thresh)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # image ,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # cv2.imshow('image',image)
    # x,y,w,h = cv2.boundingRect(cnt)
    #output

    output = cv2.bitwise_and(frame , hsv , mask= mask)
    area = np.shape(output)[0]*np.shape(output)[1]
    no_red = cv2.countNonZero(mask)
    print('no_red',no_red , 'Area' , 0.1*area)
    if int(no_red) > 15000 and  int(no_red) >int(0.1*area):
        count = count+1
        color = 0,0,255
        if count >=Threshold:
            print('Counter',count)
            cv2.putText(frame,"FIRE!!", (100,100), cv2.FONT_HERSHEY_SIMPLEX,3,color,3,cv2.LINE_AA)
            print ('Fire detected')
    else:
        count = 0
    #print('Output:',output)
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Output" ,output)
    cv2.imshow("Fire region :" , frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()
