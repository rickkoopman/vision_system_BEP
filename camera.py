import cv2
import numpy as np
import pickle

from gstreamer_pipeline import gstreamer_pipeline


class Camera:
    def __init__(self, index=0, load_path=None, gstreamer=False):
        if gstreamer:
            self.cap = cv2.VideoCapture(
                gstreamer_pipeline(
                    sensor_id=index,
                    capture_width=3264,
                    capture_height=1848,
                    display_width=960,
                    display_height=540,
                    framerate=28,
                ),
                cv2.CAP_GSTREAMER,
            )
        else:
            self.cap = cv2.VideoCapture(index)

        if load_path is not None:
            with open(load_path, "rb") as f:
                mtx, dist, newcameramtx, roi = pickle.load(f)
                self.camera_matrix = mtx
                self.distortion = dist
                self.new_camera_matrix = newcameramtx
                self.region_of_interest = roi
        else:
            self.camera_matrix = None
            self.distortion = None
            self.new_camera_matrix = None
            self.region_of_interest = None

    def __del__(self):
        self.cap.release()

    def gstreamer_set_values(
        self,
        sensor_id=0,
        capture_size=(3264, 1848),
        display_size=(960, 540),
        framerate=28,
    ):
        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(
                sensor_id=sensor_id,
                capture_width=capture_size[0],
                capture_height=capture_size[1],
                display_width=display_size[0],
                display_height=display_size[1],
                framerate=framerate,
            ),
            cv2.CAP_GSTREAMER,
        )

    def read(self):
        _, frame = self.cap.read()
        return frame

    def calibrate(self, checkerboard_size=(9, 6), save_to_path=None):
        # Get pictures of checkerboard

        pictures = []

        while True:
            frame = self.read()
            cv2.imshow("Calibration", frame)

            keycode = cv2.waitKey(10) & 0xFF
            if keycode in [27, ord("q")]:
                break
            elif keycode in [32, ord("f")]:
                print(f"Taking picture ({len(pictures) + 1})")
                pictures.append(frame)

        # Find checkerboard corners

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : checkerboard_size[1], 0 : checkerboard_size[0]
        ].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for picture in pictures:
            gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                cv2.drawChessboardCorners(picture, checkerboard_size, corners2, ret)
                cv2.imshow("corners", picture)
                cv2.waitKey()

        cv2.destroyAllWindows()

        # Calculate values

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        w, h = pictures[0].shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        self.camera_matrix = mtx
        self.distortion = dist
        self.new_camera_matrix = newcameramtx
        self.region_of_interest = roi

        # Save values

        if save_to_path is not None:
            with open(save_to_path, "wb") as f:
                pickle.dump(
                    [
                        self.camera_matrix,
                        self.distortion,
                        self.new_camera_matrix,
                        self.region_of_interest,
                    ],
                    f,
                )


if __name__ == "__main__":
    calibrate = True

    location = './calibration_left.pkl'
    camera = Camera(0, gstreamer=True, load_path=None if calibrate else location)
    if calibrate:
        camera.calibrate(save_to_path=location)
    print(camera.camera_matrix)
    print(camera.new_camera_matrix)