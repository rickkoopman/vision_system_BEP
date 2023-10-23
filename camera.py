import cv2
import numpy as np
import pickle
import os
from gstreamer_pipeline import gstreamer_pipeline


class Camera:
    def __init__(
        self,
        sensor_id=0,
        capture_size=(3264, 1848),
        display_size=(960, 540),
        framerate=28,
    ):
        capture_width, capture_height = capture_size
        display_width, display_height = display_size

        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(
                sensor_id=sensor_id,
                capture_width=capture_width,
                capture_height=capture_height,
                display_width=display_width,
                display_height=display_height,
                framerate=framerate,
            ),
            cv2.CAP_GSTREAMER,
        ) 

        self.camera_roll = []

        self.camera_model = {}

    def take_picture(self, camera_roll=False):
        ret, img = self.cap.read()
        if ret == False:
            raise RuntimeError("Could not read from VideoCapture")
        self.last_image = img
        return self.last_image

    def show_last_image(self, waitKey=False):
        cv2.imshow("image", self.last_image)
        if waitKey:
            cv2.waitKey()

    def film(self, func=None):
        while True:
            self.take_picture()
            self.show_last_image()

            keyCode = cv2.waitKey(10) & 0xFF
            if keyCode == 27 or keyCode == ord("q"):
                break

        cv2.destroyAllWindows()

    def take_pictures(self, camera_roll=True):
        pictures = []

        while True:
            self.take_picture(camera_roll=False)
            self.show_last_image()

            keyCode = cv2.waitKey(10) & 0xFF
            if keyCode == 27 or keyCode == ord("q"):
                break
            elif keyCode == ord("f"):
                print(f"Taking Picture {len(pictures) + 1}")

                pictures.append(self.last_image)
                if camera_roll:
                    self.camera_roll.append(self.last_image)

        cv2.destroyAllWindows()
        return pictures

    def clear_camera_roll(self):
        self.camera_roll = []

    def show_camera_roll(self):
        for image in self.camera_roll:
            cv2.imshow("camera roll", image)
            keyCode = cv2.waitKey()
            if keyCode == 27 or keyCode == ord("q"):
                break

    def get_calibration_points(self, checkerboard_pictures=None, chessboard_size=(9, 6), block_size=23):

        if checkerboard_pictures == None:
            checkerboard_pictures = self.take_pictures(camera_roll=False)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # figure out what this code does
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        objp = objp * block_size

        objpoints = []
        imgpoints = []

        for image in checkerboard_pictures:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret == True:
                # figure out what objp is/does
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
                cv2.imshow('image', image)
                cv2.waitKey()

        cv2.destroyAllWindows()

        image_size = checkerboard_pictures[0].shape[:2]

        return objpoints, imgpoints, image_size
    
    def calibrate_compute(self, objpoints, imgpoints, image_size, chessboard_size=(9, 6), block_size=23):
        height, width = image_size

        ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (width, height), 1, (width, height))

        self.camera_model = dict([
            ('mtx', camera_matrix),
            ('new_mtx', new_camera_matrix),
            ('roi', roi),
            ('dist', dist),
            ('rvecs', rvecs),
            ('tvecs', tvecs)
        ])

        return ret, camera_matrix, new_camera_matrix, dist, rvecs, tvecs, roi

    def calibrate(self, chessboard_size=(9, 6), block_size=23):
        
        objpoints, imgpoints, image_size = self.get_calibration_points(chessboard_size=chessboard_size, block_size=block_size)
        
        ret, camera_matrix, new_camera_matrix, dist, rvecs, tvecs, roi = self.calibrate_compute(objpoints, imgpoints, image_size)

        print('normal camera matrix:')
        print(camera_matrix)

        print('new camera matrix:')
        print(new_camera_matrix)

        # calculate error

        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        mean_error /= len(objpoints)
        print(f'total error: {mean_error}')

    def dump_camera_model(self, path):
        with open(f'{path}.pkl', 'wb') as f:
            pickle.dump(self.camera_model, f)

    def load_camera_model(self, path):
        with open(f'{path}.pkl', 'rb') as f:
            self.camera_model = pickle.load(f)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()