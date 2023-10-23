import numpy as np
import cv2

from camera import Camera
from stereo import Stereo
from disparity import Disparity

if __name__ == '__main__':

    img_l = cv2.imread('./middlebury/data/octogons1/im0.png')
    img_r = cv2.imread('./middlebury/data/octogons1/im1.png')

    # stereo = Stereo()
    # stereo.calibrate()
    # stereo.dump_stereo_model('😀')

    disparity = Disparity(3, 15)
    disparity.load_images(img_l, img_r)
    disparity.compute()
    disparity.show()
