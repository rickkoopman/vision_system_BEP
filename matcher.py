import numpy as np
import cv2
import matplotlib.pyplot as plt


class Matcher:
    def __init__(self, num_disparities=16 * 8, block_size=15, B=None, f=None):
        self.__num_disparities = num_disparities
        self.__block_size = block_size
        self.__lambda = 8000
        self.__sigma = 1.5

        self.__left_matcher = cv2.StereoBM.create(
            numDisparities=self.__num_disparities,
            blockSize=self.__block_size,
        )
        self.__right_matcher = cv2.ximgproc.createRightMatcher(self.__left_matcher)

        self.__wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.__left_matcher)
        self.__wls_filter.setLambda(self.__lambda)
        self.__wls_filter.setSigmaColor(self.__sigma)

        self.__left_image = None
        self.__right_image = None
        self.__disparity = None
        self.__history = []

    def set_wls_parameters(self, lmbda, sigma):
        self.__wls_filter.setLambda(lmbda)
        self.__wls_filter.setSigma(sigma)

    def load_images(self, left_image, right_image, size=(960, 540), blur_size=3):
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        left_image = cv2.resize(left_image, size)
        right_image = cv2.resize(right_image, size)

        left_image = cv2.medianBlur(left_image, blur_size)
        right_image = cv2.medianBlur(right_image, blur_size)

        self.__left_image = left_image
        self.__right_image = right_image

    def compute(self, wls_filter=False, remove_outliers=False):
        if self.__left_image is None or self.__right_image is None:
            print("No images loaded yet")
            return

        disparity = self.__left_matcher.compute(self.__left_image, self.__right_image)

        if wls_filter:
            disparity_left = disparity
            disparity_right = self.__right_matcher.compute(
                self.__right_image, self.__left_image
            )
            disparity = self.__wls_filter.filter(
                disparity_left, self.__left_image, disparity_map_right=disparity_right
            )

        disparity = np.where(disparity < 0, 0, disparity)[
            :, self.__num_disparities :
        ]

        if remove_outliers:
            std = np.nanstd(disparity)
            mean = np.nanmean(disparity)

            dist = np.abs(disparity - mean)
            disparity = np.where(dist < std * 2, disparity, 0)

        self.__disparity = disparity / self.__num_disparities

        self.__history.append(self.__disparity)
        while len(self.__history) > 1:
            self.__history.pop(0)

        return self.__disparity
    
    ## TODO

    @property
    def disparity(self):
        return self.__disparity
    
    @property
    def normalized_disparity(self):
        disp = self.__disparity
        return (disp - disp.min()) / (disp.max() - disp.min())
    
    @property
    def mean_disparity_over_time(self):
        disp = np.array(self.__history).mean(axis=0)
        return (disp - disp.min()) / (disp.max() - disp.min())
    
    def plot_disparity(self):
        if self.__disparity is None:
            print("Disparity has not yet been computed")
        else:
            disp = self.__disparity
            disp = np.where(disp < 0, np.nan, disp)

            fig, axs = plt.subplots(1, 2)
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            axs[0].imshow(disp)
            axs[1].imshow(self.__left_image)

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.show()

    @property
    def depth(self):
        return B * f / self.disparity
