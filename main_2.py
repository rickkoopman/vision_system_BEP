import numpy as np
import cv2

from camera import Camera
# from stereo import Stereo
# from disparity import Disparity

if __name__ == '__main__':

    left = Camera(sensor_id=0)
    left.calibrate