import numpy as np
import cv2
import pickle
from camera import Camera

class Stereo:
    def __init__(self, capture_size=(3264, 1848), display_size=(960, 540), framerate=28):

        self.left = Camera(sensor_id=0, capture_size=capture_size, display_size=display_size, framerate=framerate)
        self.right = Camera(sensor_id=1, capture_size=capture_size, display_size=display_size, framerate=framerate)

        self.stereo_model = {}
        self.stereo = cv2.StereoBM_create()

    def take_pictures(self):
        pictures = []

        while True:
            left = self.left.take_picture()
            right = self.right.take_picture()

            cv2.imshow('film', np.hstack((left, right)))

            keyCode = cv2.waitKey(10) & 0xff
            if keyCode == 27 or keyCode == ord('q'):
                break
            elif keyCode == ord('f'):
                print(f"Taking Picture {len(pictures) + 1}")
                pictures.append((left, right))

        return pictures

    def get_calibration_points(self, checkerboard_pictures=None, chessboard_size=(9, 6), block_size=23):
        if checkerboard_pictures == None:
            checkerboard_pictures = self.take_pictures()
            
        left_pictures = [left for left, _ in checkerboard_pictures]
        right_pictures = [right for _, right in checkerboard_pictures]

        objpoints, imgpoints_left, image_size = self.left.get_calibration_points(left_pictures)
        _, imgpoints_right, _ = self.right.get_calibration_points(right_pictures)

        return objpoints, imgpoints_left, imgpoints_right, image_size

    def calibrate_compute(self, objpoints, imgpoints_left, imgpoints_right, image_size, chessboard_size=(9, 6), block_size=23):

        self.left.calibrate_compute(objpoints, imgpoints_left, image_size, chessboard_size, block_size)
        self.right.calibrate_compute(objpoints, imgpoints_right, image_size, chessboard_size, block_size)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        width, height = image_size
        stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC

        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, 
            imgpoints_left, 
            imgpoints_right, 
            self.left.camera_model['mtx'],
            self.left.camera_model['dist'],
            self.right.camera_model['mtx'],
            self.right.camera_model['dist'],
            (width, height),
            criteria=criteria,
            flags=stereo_calibration_flags
        )

        return ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F


    def calibrate(self, chessboard_size=(9, 6), block_size=23):
        objpoints, imgpoints_left, imgpoints_right, image_size = self.get_calibration_points(chessboard_size=chessboard_size, block_size=block_size)

        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = self.calibrate_compute(
            objpoints, 
            imgpoints_left, 
            imgpoints_right, 
            image_size,
            chessboard_size,
            block_size,
        )
        
        self.stereo_model = dict([
            ('mtx_left', mtx_left),
            ('dist_left', dist_left),
            ('mtx_right', mtx_right),
            ('dist_right', dist_right),
            ('rotation', R),
            ('translation', T),
            ('essential', E),
            ('fundamental', F)
        ])

    def dump_stereo_model(self, path):
        with open(f'{path}.pkl', 'wb') as f:
            pickle.dump(self.stereo_model, f)

    def load_stereo_model(self, path):
        with open(f'{path}.pkl', 'rb') as f:
            self.stereo_model = pickle.load(f)

    def disparity_simple(self, num_disparities=16, block_size=15):
        while True:
            left = self.left.take_picture()
            right = self.right.take_picture()

            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            disparity = self.stereo.compute(left_gray, right_gray)
            disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            cv2.imshow('disparity', disparity_normalized)

            keyCode = cv2.waitKey(10) & 0xff
            if keyCode == 27 or keyCode == ord('q'):
                break

    def disparity_complex(self):

        mtx_left = self.stereo_model['mtx_left']
        mtx_right = self.stereo_model['mtx_right']
        dist_left = self.stereo_model['dist_left']
        dist_right = self.stereo_model['dist_right']
        rotation = self.stereo_model['rotation']
        translation = self.stereo_model['translation']

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.rectifyStereo(mtx_left, dist_left, mtx_right, dist_right, rotation, translation)

        # depth = cv2.reprojectImageTo3D()


    def overlap(self):
        while True:
            left = self.left.take_picture()
            right = self.right.take_picture()

            combine = cv2.addWeighted(left, 0.5, right, 0.5, 0)

            keyCode = cv2.waitKey(10) & 0xff
            if keyCode == 27 or keyCode == ord('q'):
                break

    def film(self):
        while True:
            left = self.left.take_picture()
            right = self.right.take_picture()

            cv2.imshow('film', np.hstack((left, right)))

            keyCode = cv2.waitKey(10) & 0xff
            if keyCode == 27 or keyCode == ord('q'):
                break