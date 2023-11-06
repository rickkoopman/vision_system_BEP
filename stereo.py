import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from camera import Camera
from disparity import Disparity


class Stereo:
    def __init__(self):
        self.__left_camera = Camera(index=0, gstreamer=True)
        self.__right_camera = Camera(index=1, gstreamer=True)

    def gstreamer_set_values(
        self,
        capture_size=(3264, 1848),
        display_size=(960, 540),
        framerate=28,
    ):
        self.__left_camera.gstreamer_set_values(
            sensor_id=0,
            capture_size=capture_size,
            display_size=display_size,
            framerate=framerate,
        )
        self.__right_camera.gstreamer_set_values(
            sensor_id=1,
            capture_size=capture_size,
            display_size=display_size,
            framerate=framerate,
        )

    def read(self):
        left_image = self.__left_camera.read()
        right_image = self.__right_camera.read()
        return left_image, right_image


if __name__ == "__main__":
    stereo = Stereo()
    matcher = Disparity(num_disparities=16 * 8, block_size=13)

    while True:
        left, right = stereo.read()
        matcher.load_images(left, right, blur_size=3)
        disparity = matcher.compute(wls_filter=False)
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imshow('disparity', disparity)
        # cv2.imshow("image", np.hstack((left, right)))
        keycode = cv2.waitKey(10) & 0xFF
        if keycode in [27, ord("q")]:
            break
    cv2.destroyAllWindows()

    # rick
