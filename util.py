import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial.transform import Rotation
from time import sleep

from lib.cars import Camera, Connection, Motors

red_marker = 6
green_marker = 8


class CameraOperator:
    def __init__(self, connection, display=False, draw_markers=False, draw_axis=False):
        self._camera = Camera(connection)
        self._mapx = np.load('calibration/mapx.npy')
        self._mapy = np.load('calibration/mapy.npy')
        self._camera_matrix = np.load('calibration/camera_matrix.npy')
        self._dist_coeffs = np.load('calibration/dist_coeffs.npy')
        self._marker_side = 42
        self._markers_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self._detector_params = cv2.aruco.DetectorParameters_create()
        self._detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self._display = display
        self._draw_markers = draw_markers
        self._draw_axis = draw_axis

    def flash(self):
        self._camera.flash_on()
        sleep(0.1)
        self._camera.flash_off()

    def flash_on(self):
        self._camera.flash_on()

    def flash_off(self):
        self._camera.flash_off()

    def detect_markers(self):
        """
        Takes a photo and returns transformation and rotation of all detected markers.
        """

        # Frames tend to buffer.
        for _ in range(5):
            self._camera.get_frame()
        img = self._camera.get_frame()

        # Undistort image.
        img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_LINEAR)

        corners, ids, _ = cv2.aruco.detectMarkers(img, self._markers_dictionary, None, None,
                                                  self._detector_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self._marker_side,
                                                              self._camera_matrix,
                                                              self._dist_coeffs)

        if self._display:
            if self._draw_markers:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)

            if self._draw_axis and ids is not None:
                cv2.aruco.drawAxis(img, self._camera_matrix, self._dist_coeffs, rvecs[0], tvecs[0],
                                   100)
            cv2.imshow('robot_view', img)
            cv2.waitKey(1)

        if ids is not None:
            ids = ids.reshape(-1)
            tvecs = tvecs.reshape(-1, 3)
            rvecs = Rotation.from_rotvec(rvecs.reshape(-1, 3)).as_euler('xyz')

        return ids, rvecs, tvecs

    def detect_basket_color(self):
        """ Returns borders of the basket and marker id based on the color. """

        img = self._camera.get_frame()

        hsv_photo = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_red_1 = cv2.inRange(hsv_photo, np.array([0, 100, 50]), np.array([10, 255, 255]))
        mask_red_2 = cv2.inRange(hsv_photo, np.array([160, 100, 50]), np.array([180, 255, 255]))
        mask_red = (mask_red_1 + mask_red_2) // 255
        mask_green = cv2.inRange(hsv_photo, np.array([40, 60, 60]), np.array([80, 255, 255])) // 255

        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size))
        mask_red = convolve(mask_red, kernel, mode='constant', cval=0) // (kernel_size ** 2)
        mask_green = convolve(mask_green, kernel, mode='constant', cval=0) // (kernel_size ** 2)

        if mask_green.sum() > mask_red.sum():
            marker = green_marker
            mask = mask_green
        else:
            marker = red_marker
            mask = mask_red

        match = np.where(mask.sum(axis=0) > 0)[0]
        return marker, match[0], match[-1]


class RobotBase:
    def __init__(self, connection, display=False, draw_markers=False, draw_axis=False):
        self.connection = connection
        self.motors = Motors(connection=connection)
        self.camera_operator = CameraOperator(connection, display, draw_markers, draw_axis)


def run_robot(robot_class, display=True, **kwargs):
    if display:
        cv2.namedWindow('robot_view')

    connection = Connection()
    robot = robot_class(connection, display=display, **kwargs)
    robot.run()
