import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from camera import Camera
from stereo import Stereo
from disparity import Disparity


def showcase_disparity():

    stereo = Disparity(num_disparities=16*8, block_size=9)

    image_names = os.listdir('./middlebury/data')
    random.shuffle(image_names)
    
    fig = plt.figure()
    fig.tight_layout()

    for i, image_name in enumerate(image_names[:3]):
        left_image = cv2.imread(f"./middlebury/data/{image_name}/left.png")
        right_image = cv2.imread(f"./middlebury/data/{image_name}/right.png")

        stereo.load_images(left_image, right_image)
        stereo.compute(wls_filter=True, remove_outliers=False)
        
        ax1 = plt.subplot(2, 3, i + 1)
        ax2 = plt.subplot(2, 3, i + 4)
        ax1.imshow(stereo.disparity, cmap='plasma')
        ax2.imshow(left_image)
    
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def main():
    showcase_disparity()

if __name__ == "__main__":
    main()