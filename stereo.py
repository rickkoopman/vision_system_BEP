import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from camera2 import Camera
from disparity import Disparity

class Stereo:
    def __init__(self, capture_size=(3264, 1848), display_size=(960, 540), framerate=28):

        self.__left_camera = Camera(sensor_id=0, capture_size=capture_size, display_size=display_size, framerate=framerate)
        self.__right_camera = Camera(sensor_id=1, capture_size=capture_size, display_size=display_size, framerate=framerate)

    def read(self):
        left_image = self.__left_camera.read()
        right_image = self.__right_camera.read()
        return left_image, right_image
    

if __name__ == "__main__":
    stereo = Stereo()
    matcher = Disparity(num_disparities=16 * 8, block_size=9)

    # left_image = cv2.imread(f'./middlebury/data/chess1/im0.png')
    # right_image = cv2.imread(f'./middlebury/data/chess1/im1.png')

    matcher.load_images(*stereo.read(), size=(960, 540), blur_size=3)
    disparity = matcher.compute(wls_filter=True, remove_outliers=False)

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(disparity, cmap='plasma')
    plt.show()