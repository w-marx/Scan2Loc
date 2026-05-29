import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gathered_robot_data import assert_intrinsic_mat
import numpy as np

class HeadsetData:
    def __init__(
            self,
            bgr_image_s:np.ndarray,
            intrinsic_camera_matrix:np.ndarray,
            distortion_coefficients:list[float],
    ):
        """
        :param bgr_image_s: BGR images as an NxHxWx3-uint8 numpy array
        :param intrinsic_camera_matrix: Intrinsic camera matrix as a 3x3 numpy array
        :param distortion_coefficients: Distortion coefficients as a list of floats
        """
        assert bgr_image_s.ndim == 4, f"Wrong shape of BGR images {bgr_image_s.shape}"
        assert bgr_image_s.shape[0] > 0, f"No BGR images {bgr_image_s.shape}"
        assert bgr_image_s.dtype == np.uint8, f"Wrong dtype for bgr images {bgr_image_s.dtype}"
        self._bgr_image_s = bgr_image_s


        assert assert_intrinsic_mat(intrinsic_camera_matrix, self._bgr_image_s[0])
        self._intrinsic_camera_matrix = intrinsic_camera_matrix

        assert isinstance(distortion_coefficients, list) and len(distortion_coefficients) > 0
        self._distortion_coefficients = distortion_coefficients


    @classmethod
    def from_vrs_file(cls, file_location):
        """
        This function takes a .vrs file and creates a HeadsetData instance
        It undistorts the images by taking the camera-rgb - fisheye camera and transforming it to a pinhole camera
        It makes the assumption that the fisheye camera focal length is the average of fx and fy.
        The distortion coefficients are assumed to be 0

        :param file_location: The location of the .vrs file including the filename
        :return: an NxWxHx3-uint8 RGB image array, the camera intrinsic matrix, the camera distortion coefficients
        """
        if not os.path.isfile(file_location):
            raise FileNotFoundError(f"No .vrs file found at {file_location}")

        provider = data_provider.create_vrs_data_provider(file_location)
        stream_id = provider.get_stream_id_from_label("camera-rgb")
        cam_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
        img_width, img_height = cam_calib.get_image_size()
        focal_length = (cam_calib.get_focal_lengths()[0] + cam_calib.get_focal_lengths()[1]) / 2
        pinhole = calibration.get_linear_camera_calibration(image_width=img_width, image_height=img_height,focal_length=focal_length, label="camera-rgb")

        bgr_hxw_imgs = []
        for i in range(0, provider.get_num_data(stream_id)):
            image_data = provider.get_image_data_by_index(stream_id, i)[0].to_numpy_array()
            undistorted_image = calibration.distort_by_calibration(arraySrc=image_data, dstCalib=pinhole,srcCalib=cam_calib)
            bgr_image = cv2.cvtColor(np.array(undistorted_image), cv2.COLOR_RGB2BGR)
            bgr_hxw_imgs.append(cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE))

        fx, fy = pinhole.get_focal_lengths()
        cx, cy = pinhole.get_principal_point()
        mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return cls(bgr_image_s=np.array(bgr_hxw_imgs),intrinsic_camera_matrix=mtx, distortion_coefficients=[0,0,0,0,0])

    @property
    def bgr_image_s(self):
        return self._bgr_image_s

    @property
    def intrinsic_camera_matrix(self):
        return self._intrinsic_camera_matrix

    @property
    def distortion_coefficients(self):
        return self._distortion_coefficients