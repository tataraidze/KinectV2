import numpy as np
import PyKinectV2
import PyKinectRuntime
import cv2
import os
import time
import const
import shelve

class KinectV2(object):

    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                 PyKinectV2.FrameSourceTypes_BodyIndex |
                                                 PyKinectV2.FrameSourceTypes_Depth)

        #load calibration results
        self.rgbCamera = shelve.open(const.rgbCameraIntrinsic)
        self.rgb_Fx = self.rgbCamera['camera_matrix'][0,0]
        self.rgb_Fy = self.rgbCamera['camera_matrix'][1,1]
        self.rgb_Cx = self.rgbCamera['camera_matrix'][0,2]
        self.rgb_Cy = self.rgbCamera['camera_matrix'][1,2]

        self.irCamera = shelve.open(const.irCameraIntrinsic)
        self.depth_Fx = self.irCamera['camera_matrix'][0,0]
        self.depth_Fy = self.irCamera['camera_matrix'][1,1]
        self.depth_Cx = self.irCamera['camera_matrix'][0,2]
        self.depth_Cy = self.irCamera['camera_matrix'][1,2]

        self.stereoCamera = shelve.open(const.rgbToIR)
        distanceFile = shelve.open(const.distanceErrorFunction)
        self.depthErrorFunction = dict(zip(distanceFile['x'], distanceFile['y']))

        self.numberOfPicture = 0

    def close(self):
        self.kinect.close()


    def takePicture(self):
        #get frames from Kinect
        colorFrame = self.kinect.get_last_color_frame()
        depthFrame = self.kinect.get_last_depth_frame()
        bodyIndexFrame = self.kinect.get_last_body_index_frame()

        #reshape to 2-D space
        colorFrame = colorFrame.reshape((const.rgb_image_size[0],const.rgb_image_size[1],4))
        depthFrame = depthFrame.reshape(const.ir_image_size)
        bodyIndexFrame = bodyIndexFrame.reshape(const.ir_image_size)

        #compensate  lens distortion
        colorFrame = cv2.undistort(colorFrame,self.rgbCamera['camera_matrix'], self.rgbCamera['dist_coefs'])
        depthFrame = cv2.undistort(depthFrame,self.irCamera['camera_matrix'], self.irCamera['dist_coefs'])
        bodyIndexFrame = cv2.undistort(bodyIndexFrame,self.irCamera['camera_matrix'], self.irCamera['dist_coefs'])

        distanceToBody = np.zeros(const.ir_image_size,np.uint16)
        distanceToBody[bodyIndexFrame != const.indexForBackground] = depthFrame[bodyIndexFrame != const.indexForBackground]
        #compensate systematic errors of Kinect depth sensor
        for i in range(0,distanceToBody[:,0].__len__()):
            for j in range(0,distanceToBody[0,:].__len__()):
                if distanceToBody[i,j] not in self.depthErrorFunction:
                    print(str(distanceToBody[i,j]) + "is not in range of the depth error function!")
                else:
                    distanceToBody[i,j] = distanceToBody[i,j] - self.depthErrorFunction[distanceToBody[i,j]]

        #combine depth, RGB and bodyIndexFrame
        combinedImage = self.__align__(colorFrame,distanceToBody)

        self.__saveData__(colorFrame, depthFrame.reshape(const.ir_image_size), combinedImage, distanceToBody.reshape(const.ir_image_size))
        self.numberOfPicture = self.numberOfPicture + 1

        return combinedImage

    def newExperiment(self, folderName=[]):
        if not folderName:
            folderName = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        self.folderName = const.pictureFolder + folderName + '/'
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)
        self.numberOfPicture = 1


    def __saveData__(self,colorFrame, depthArray, combinedImage, distanceToBody):
        cv2.imwrite(self.folderName + str(self.numberOfPicture) + '_rgb.jpg',colorFrame)
        cv2.imwrite(self.folderName + str(self.numberOfPicture) + '_combined.jpg',combinedImage)
        depthArray.tofile(self.folderName + str(self.numberOfPicture) + '_depth.dat')
        distanceToBody.tofile(self.folderName +str(self.numberOfPicture) + '_distanceToBody.dat')

    #colorize depth frame
    def __align__(self,colorFrame,depthFrame):
        combinedImage = np.zeros((const.ir_image_size[0],const.ir_image_size[1],3))
        depthFrame = depthFrame/1000 #from mm to meters

        # From Depth Map to Point Cloud
        #book "Hacking the Kinect" by Jeff Kramer P. 130, http://gen.lib.rus.ec/book/index.php?md5=55a9155e10b1d5bb92811f69cb15f127
        worldCoordinates = np.zeros((np.prod(const.ir_image_size),3))
        i = 0
        for depthX in range(1,depthFrame.shape[1]):
            for depthY in range(1, depthFrame.shape[0]):
                z = depthFrame[depthY,depthX]
                if (z > 0):
                    worldCoordinates[i,0] = z*(depthX-self.depth_Cx)/self.depth_Fx #x
                    worldCoordinates[i,1] = z*(depthY-self.depth_Cy)/self.depth_Fy #y
                    worldCoordinates[i,2] = z #z
                i = i + 1

        #Projecting onto the Color Image Plane, Hacking the Kinect, P. 132
        worldCoordinates = np.dot(worldCoordinates, self.stereoCamera['R'].T) + self.stereoCamera['T'].T
        rgbX = np.round(worldCoordinates[:,0]*self.rgb_Fx/worldCoordinates[:,2]+self.rgb_Cx)
        rgbY = np.round(worldCoordinates[:,1]*self.rgb_Fy/worldCoordinates[:,2]+self.rgb_Cy)

        #colorize depth image
        i=0
        for depthX in range(1,depthFrame.shape[1]):
            for depthY in range(1, depthFrame.shape[0]):
                if ((rgbX[i] >= 0) & (rgbX[i] < const.rgb_image_size[1]) & (rgbY[i] >= 0) & (rgbY[i] < const.rgb_image_size[0])):
                    combinedImage[depthY,depthX,0] = colorFrame[rgbY[i],rgbX[i]][0]
                    combinedImage[depthY,depthX,1] = colorFrame[rgbY[i],rgbX[i]][1]
                    combinedImage[depthY,depthX,2]= colorFrame[rgbY[i],rgbX[i]][2]
                i = i + 1

        combinedImage = np.uint8(combinedImage)
        return combinedImage






