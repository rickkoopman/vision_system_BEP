import numpy as np
import cv2


def color2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class Disparity:
    def __init__(self, num_disparities=3, block_size=15):
        self.stereo = cv2.StereoBM_create()
        self.left = None
        self.right = None
        self.disparity = None

        self.configuration = {
            num_disparities: num_disparities * 16,
            block_size: block_size,
        }

    def load_images(self, left, right):
        size = (960, 540)

        left = cv2.resize(left, size)
        right = cv2.resize(right, size)

        self.left = color2gray(left)
        self.right = color2gray(right)

    def set_block_size(self, block_size):
        self.configuration["block_size"] = block_size
        self.stereo.setBlockSize(block_size)

    def set_num_disparities(self, num_disparities):
        self.configuration["num_disparities"] = num_disparities
        self.stereo.setNumDisparities(16 * num_disparities)

    def compute(self, wls=False):
        self.disparity = self.stereo.compute(self.left, self.right)

    def show_images(self):
        if (self.left is not None) and (self.right is not None):
            cv2.imshow("images", np.hstack((self.left, self.right)))
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("No images loaded yet")

    def show(self):
        if self.disparity is not None:
            normalized = cv2.normalize(
                self.disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
            )
            cv2.imshow(
                f"ND: {self.configuration['num_disparities']} | BS: {self.configuration['block_size']}",
                normalized,
            )
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("No disparity calculated yet")
