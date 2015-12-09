__author__ = 'Alexander'
import time
import KinectV2
import cv2

kinect = KinectV2.KinectV2()#initialize kinect sensor
kinect.newExperiment('test')#create folder for the experiment, default name timestamp "%Y-%m-%d_%H-%M-%S"
time.sleep(2)# we cannot get reliable  data from kinect during first moments

while(1):
    combinedImage = kinect.takePicture()
    cv2.imshow('Combined',combinedImage)
    if (cv2.waitKey(1) == 27):
        cv2.destroyAllWindows()
        kinect.close()
        break


