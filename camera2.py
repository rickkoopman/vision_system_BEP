import cv2
import numpy as np


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


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

    def __del__(self):
        self.cap.release()

    def read(self):
        _, frame = self.cap.read()
        return frame
