import numpy as np
import cv2

from camera import Camera
from stereo import Stereo
from disparity import Disparity

if __name__ == '__main__':

    img_l = cv2.imread('./middlebury/data/octogons1/im0.png')
    img_r = cv2.imread('./middlebury/data/octogons1/im1.png')

    disparity = Disparity()
    disparity.set_num_disparities(3)
    disparity.set_block_size(15)

    disparity.load_images(img_l, img_r)
    # disparity.show_images()

    disparity.compute(wls=True)
    disparity.show()
