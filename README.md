# KinectV2

The project enables to calibrate KinectV2. You can read more [here](http://habrahabr.ru/post/272629/) (in Russian).

#### Dependencies
1. OpenCV 3.0 with Python 3 support
2. [PyKinect2](https://github.com/Kinect/PyKinect2)

#### File descriptions
* const.py - constants: paths, pattern size, e.t.c.
* takePictures.py - calibration data collecting
* calibrate.py -  single camera calibration (RGB or IR/depth)
* testDistortion.py - check whether distortion was removed or not
* stereoCalibrate.py - stereo camera calibration finds transformation |R,T| between depth camera space and RGB camera space
* depthCalibration.py - depth sensor calibration
* KinectV2.py - wrapper to PyKinect2 compensates distorion and systematic  error of depth sensor; combines depth, RGB and bodyIndex frames; saves data
* sample.py - small example of using KinectV2.py 

#### How to use
1. Change constants in const.py
2. Collect data for geometric calibration by takePictures.py (20-30 images of object in different positions)
3. Calibrate RGB camera by calibrate.py
4. Calibrate IR/depth camera by calibrate.py
5. Find |R,T| transformation between depth camera space and RGB camera space by stereoCalibrate.py
6. Collect data for depth calibration by takePictures.py ()
7. Use sample.py to get corrected data 

Support for this work was provided by the Russian Science Foundation, project #15-19-30012. 
You can find more about the project [here] (http://www.rslab.ru/russian/project/rsf2/)
