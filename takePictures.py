import cv2
import PyKinectV2
import PyKinectRuntime
import numpy as np
import const
import time

def ir_frame_to_jpg(IRFrame):
    IRFrame = IRFrame.reshape((const.ir_image_size))
    IRFrame = np.uint8(IRFrame/256)

    jpgIRFrame = np.zeros((const.ir_image_size[0],const.ir_image_size[1],3), np.uint8)
    jpgIRFrame[:,:,0]  =  IRFrame
    jpgIRFrame[:,:,1]  =  IRFrame
    jpgIRFrame[:,:,2]  =  IRFrame
    return jpgIRFrame

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Infrared |
                                         PyKinectV2.FrameSourceTypes_Depth)
i = 0
mytime = time.time()

redAlert = np.zeros((const.ir_image_size[0],const.ir_image_size[1],3),np.uint8)
redAlert[:,:,2] = 255
#cv2.namedWindow('IR', cv2.WINDOW_NORMAL)
while(1):
    mytime = time.time()

    #play video
    #while(time.time() - mytime < 5):#wait 5 sec. it is time to change an object pose
    while(cv2.waitKey(1) != 27):#wait ESC press
        cv2.imshow('IR',ir_frame_to_jpg(kinect.get_last_infrared_frame()))

    #save data
    cv2.imshow('IR',redAlert)
    cv2.waitKey(1)

    IRFrame = kinect.get_last_infrared_frame()
    jpgIRFrame = ir_frame_to_jpg(IRFrame)
    irFilePath = const.irFolder + str(i) + '.jpg'
    cv2.imwrite(irFilePath, jpgIRFrame)

    colorFrame = kinect.get_last_color_frame()
    colorFrame = colorFrame.reshape((const.rgb_image_size[0],const.rgb_image_size[1],4))
    rgbFilePath = const.rgbFolder + str(i) + '.jpg'
    cv2.imwrite(rgbFilePath, colorFrame)

    for j in range(0,const.numberOfDepthFramesForDepthCalibration):
        time.sleep(0.03)
        depthFilePath = const.depthFolder + str(i) + '_' + str(j)  + '.npy'
        depthFrame = kinect.get_last_depth_frame()
        depthFrame = depthFrame.reshape(const.ir_image_size)
        np.save(depthFilePath, depthFrame)

    i = i + 1

