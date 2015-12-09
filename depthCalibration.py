import shelve
import numpy as np
import cv2
import const
import matplotlib.pyplot as plt

#load calibration results
rgbCamera = shelve.open(const.rgbCameraIntrinsic)
irCamera = shelve.open(const.irCameraIntrinsic)

numberOfPoints = np.prod(const.pattern_size)

distance = np.zeros((const.numberOfDistanceForDepthCalibration))
errors = np.zeros((const.numberOfDistanceForDepthCalibration,numberOfPoints))
computedDistance = np.zeros((const.numberOfDistanceForDepthCalibration, numberOfPoints))
distanceFromKinect = np.zeros((const.numberOfDistanceForDepthCalibration,const.numberOfDepthFramesForDepthCalibration,numberOfPoints))

newDepthCameraMatrix, roi=cv2.getOptimalNewCameraMatrix(irCamera['camera_matrix'],irCamera['dist_coefs'],const.ir_image_size[::-1],1,const.ir_image_size[::-1])
mapx,mapy = cv2.initUndistortRectifyMap(irCamera['camera_matrix'],irCamera['dist_coefs'],None,newDepthCameraMatrix,const.ir_image_size[::-1],5)

#create object points for one image
obj_points = np.zeros((numberOfPoints, 3), np.float32)
obj_points[:, :2] = np.indices(const.pattern_size).T.reshape(-1, 2)
obj_points *= const.square_size

cornerPoints = [0, const.pattern_size[0]-1,numberOfPoints-const.pattern_size[0],numberOfPoints-1]

for k in range(0,const.numberOfDistanceForDepthCalibration):
    irName = const.irFolder + str(k)
    rgbName = const.rgbFolder + str(k)
    irImageData =  shelve.open(irName)
    rgbImageData =  shelve.open(rgbName)

    #get undistorted coordinates of key point on IR/depth frame
    irImagePoints = np.zeros((numberOfPoints,1,2))
    irImagePoints[:,0,:] = irImageData['img_points']
    irImagePoints = cv2.undistortPoints(irImagePoints, irCamera['camera_matrix'],irCamera['dist_coefs'])

    x = irImagePoints[:,0,0] * irCamera['camera_matrix'][0,0] + irCamera['camera_matrix'][0,2]
    x = np.uint16(np.round(x+0.5))
    y = irImagePoints[:,0,1] * irCamera['camera_matrix'][1,1] + irCamera['camera_matrix'][1,2]
    y = np.uint16(np.round(y+0.5))

    #get coordinates of key point in RGB camera space
    rgbImagePoints = rgbImageData['img_points']

    rgbImagePoints = rgbImagePoints.reshape((1,48,2))
    obj_points = obj_points.reshape((1,48,3))

    retval, R, T = cv2.solvePnP(obj_points[:,cornerPoints], rgbImagePoints[:,cornerPoints], rgbCamera['camera_matrix'],
                                rgbCamera['dist_coefs'], flags = cv2.SOLVEPNP_UPNP)

    R, jacobian = cv2.Rodrigues(R)
    pointsInCameraSpace = np.dot(obj_points, R.T) + T.T
    computedDistance[k,:] = pointsInCameraSpace[0,:,2]* 1000# Z(depth) in mm

    #average distance to key points by Kinect depth sensor
    for i in range(0,const.numberOfDepthFramesForDepthCalibration):
        depthName = const.depthFolder + str(k) + '_' + str(i) + '.npy'
        depthFrame = np.load(depthName)
        depthFrame = cv2.remap(depthFrame, mapx, mapy, cv2.INTER_CUBIC)#undistort frame

        distanceFromKinect[k,i] = depthFrame[y, x]

    errors[k] = distanceFromKinect[k,1:].mean(0) - computedDistance[k]
    distance[k] = computedDistance[k].mean()

#interpolate error function
z = np.polyfit(distance, errors.mean(1), 4)
errorFunc = np.poly1d(z)

xErrorFunc = np.arange(distance.min(),distance.max(),1)
yErrorFunc = errorFunc(np.arange(distance.min(),distance.max(),1))

#save it
distanceErrorFile = shelve.open(const.distanceErrorFunction, 'n')
distanceErrorFile['x'] = xErrorFunc
distanceErrorFile['y'] = yErrorFunc
distanceErrorFile.close()

#and display
plt.ion()
plt.figure('Errors')
plt.plot(distance,errors.mean(1))
plt.plot(xErrorFunc, yErrorFunc)
plt.waitforbuttonpress()