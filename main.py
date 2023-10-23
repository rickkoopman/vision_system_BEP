from camera import Camera
from stereo import Stereo

stereo = Stereo()

if __name__ == '__main__':
    stereo.load_stereo_model('stereo_calibration')
    # stereo.calibrate()
    # stereo.dump_stereo_model('stereo_calibration')
    stereo.disparity_simple(16*6,9)