#single Kinect v2 camera calibration (RGB or IR)
#based on OpenCV example https://github.com/Itseez/opencv/blob/master/samples/python2/calibrate.py

import numpy as np
import cv2
import shelve
import os
import const
from glob import glob

isRGB = True

if (isRGB):
    img_names = glob(const.rgbFolder + '*.jpg')
    CAMERA_PATH = const.rgbCameraIntrinsic
else:
    img_names = glob(const.irFolder + '*.jpg')
    CAMERA_PATH = const.irCameraIntrinsic

#create object points
pattern_points = np.zeros((np.prod(const.pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(const.pattern_size).T.reshape(-1, 2)
pattern_points *= const.square_size

obj_points = []
img_points = []
h, w = 0, 0
for fn in img_names:
    print('processing %s...' % fn, )
    img = cv2.imread(fn, 0)
    if img is None:
        print("Failed to load", fn)
        continue

    h, w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, const.pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
    if found:
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if not found:
        print('chessboard not found')
        continue

    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    #save img_points for future stereo calibration
    img_file = shelve.open(os.path.splitext(fn)[0],'n')
    img_file['img_points'] = corners.reshape(-1, 2)
    img_file.close()

    print('ok')

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                                   img_points,
                                                                   (w, h),
                                                                   None,
                                                                   None,
                                                                   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.001),
                                                                   flags = 0)
#save calibration results
camera_file = shelve.open(CAMERA_PATH, 'n')
camera_file['camera_matrix'] = camera_matrix
camera_file['dist_coefs'] = dist_coefs
camera_file.close()

print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())
