import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from camera import Camera, CameraGStreamer
from stereo import Stereo
from matcher import Matcher


def showcase_disparity():
    stereo = Matcher(num_disparities=16 * 8, block_size=9)

    image_names = os.listdir("./middlebury/data")
    random.shuffle(image_names)

    fig = plt.figure()
    fig.tight_layout()

    for i, image_name in enumerate(image_names[:3]):
        left_image = cv2.imread(f"./middlebury/data/{image_name}/im0.png")
        right_image = cv2.imread(f"./middlebury/data/{image_name}/im1.png")

        stereo.load_images(left_image, right_image)
        stereo.compute(wls_filter=True, remove_outliers=False)

        ax1 = plt.subplot(2, 3, i + 1)
        ax2 = plt.subplot(2, 3, i + 4)
        ax1.imshow(stereo.disparity, cmap="plasma")
        ax2.imshow(left_image)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


if __name__ == "__main__":
    stereo = Stereo(gstreamer=True, capture_size=(1920, 1080), display_size=(1920, 1080), framerate=28)
    matcher = Matcher(num_disparities=16*8, block_size=11, b)
    matcher.set_wls_parameters(lmbda=8000, sigma=1.5)

    # matcher.load_images(*stereo.read(), blur_size=5)
    # matcher.compute()
    # matcher.plot()

    while True: 
        matcher.load_images(*stereo.read(), blur_size=3)
        matcher.compute(wls_filter=True)
        cv2.imshow('disparity', matcher.mean_disparity_over_time)

        keyCode = cv2.waitKey(10) & 0xff
        if keyCode in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
