#check whether undistortion works or not
import shelve
from glob import glob
import cv2
import const

isRGB = False

if isRGB:
    camera =  shelve.open(const.rgbCameraIntrinsic)
    fileNames = glob(const.rgbFolder + '*.jpg')
    size = const.rgb_image_size
else:
    camera =  shelve.open(const.irCameraIntrinsic)
    fileNames = glob(const.irFolder + '*.jpg')
    size = const.ir_image_size

newDepthCameraMatrix, roi=cv2.getOptimalNewCameraMatrix(camera['camera_matrix'],camera['dist_coefs'],size,1)
mapx,mapy = cv2.initUndistortRectifyMap(camera['camera_matrix'],camera['dist_coefs'],None,newDepthCameraMatrix,size,5)

for i in range(0,fileNames.__len__()):
    frame = cv2.imread(fileNames[i])
    frame = cv2.undistort(frame,camera['camera_matrix'], camera['dist_coefs'])
    #frame = cv2.remap(frame,mapx,mapy,cv2.INTER_LANCZOS4)

    cv2.imshow('undistort',frame)
    cv2.waitKey(0)