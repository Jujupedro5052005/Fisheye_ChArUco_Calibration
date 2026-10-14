#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np


def fisheye_flag(name):
    if hasattr(cv2.fisheye, name):
        return getattr(cv2.fisheye, name)

    if hasattr(cv2, name):
        return getattr(cv2, name)

    raise AttributeError(f"OpenCV fisheye flag not found: {name}")


class CharucoCalibrator:
    def __init__(
        self,
        aruco_dict,
        squares_vertically,
        squares_horizontally,
        square_length,
        marker_length,
        calibration_images_dir=None,
        raw_images_dir=None,
    ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)

        self.squares_vertically = squares_vertically
        self.squares_horizontally = squares_horizontally
        self.square_length = square_length
        self.marker_length = marker_length

        self.calibration_image_dir = calibration_images_dir
        self.raw_image_dir = raw_images_dir

        self.cur_dir = os.path.dirname(os.path.abspath(__file__))

        self.board = cv2.aruco.CharucoBoard(
            (squares_vertically, squares_horizontally),
            square_length,
            marker_length,
            self.aruco_dict,
        )

        self.params = cv2.aruco.DetectorParameters()

        self.K = np.zeros((3, 3), dtype=np.float64)
        self.D = None

    def image_generator(self, folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            return

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        )

        for name in sorted(os.listdir(folder_path)):
            if not name.lower().endswith(extensions):
                continue

            image = cv2.imread(
                os.path.join(folder_path, name)
            )

            if image is not None:
                yield name, image

    def calibration_images(self):
        yield from self.image_generator(
            self.calibration_image_dir
        )

    def raw_images(self):
        yield from self.image_generator(
            self.raw_image_dir
        )

    def detect_aruco_markers(
        self,
        image,
        image_name=None,
        graysale=True,
        verbose=True,
    ):
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if graysale and image.ndim == 3
            else image
        )

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.params,
            )

            marker_corners, marker_ids, rejected = (
                detector.detectMarkers(gray)
            )

            if marker_ids is not None and len(marker_ids):
                try:
                    result = detector.refineDetectedMarkers(
                        gray,
                        self.board,
                        marker_corners,
                        marker_ids,
                        rejected,
                    )

                    if result is not None and len(result) >= 2:
                        marker_corners, marker_ids = result[:2]

                except cv2.error:
                    pass

        else:
            marker_corners, marker_ids, rejected = (
                cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dict,
                    parameters=self.params,
                )
            )

            if marker_ids is not None and len(marker_ids):
                try:
                    marker_corners, marker_ids, _, _ = (
                        cv2.aruco.refineDetectedMarkers(
                            gray,
                            self.board,
                            marker_corners,
                            marker_ids,
                            rejected,
                            parameters=self.params,
                        )
                    )

                except cv2.error:
                    pass

        if marker_ids is None or len(marker_ids) == 0:
            if verbose:
                print(
                    f"No ArUco markers detected in {image_name}"
                )

            return None, None

        if verbose:
            print(
                f"{len(marker_ids)} ArUco markers detected "
                f"in {image_name}"
            )

        return marker_ids, marker_corners

    def detect_charuco_corners(
        self,
        image,
        image_name=None,
        grayscale=True,
        verbose=True,
    ):
        marker_ids, marker_corners = self.detect_aruco_markers(
            image=image,
            image_name=image_name,
            graysale=grayscale,
            verbose=verbose,
        )

        if marker_ids is None:
            return None, None

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if grayscale and image.ndim == 3
            else image
        )

        if hasattr(cv2.aruco, "interpolateCornersCharuco"):
            retval, charuco_corners, charuco_ids = (
                cv2.aruco.interpolateCornersCharuco(
                    marker_corners,
                    marker_ids,
                    gray,
                    self.board,
                )
            )

            if not retval:
                return None, None

        else:
            detector = cv2.aruco.CharucoDetector(
                self.board
            )

            charuco_corners, charuco_ids, _, _ = (
                detector.detectBoard(gray)
            )

        if charuco_ids is None or len(charuco_ids) == 0:
            return None, None

        if verbose:
            print(
                f"{len(charuco_ids)} ChArUco corners detected "
                f"in {image_name}"
            )

        return charuco_ids, charuco_corners

    def show_aruco_markers(
        self,
        image,
        image_name=None,
        window_size=(480, 480),
        verbose=True,
    ):
        out = image.copy()

        ids, corners = self.detect_aruco_markers(
            out,
            image_name=image_name,
            verbose=verbose,
        )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                out,
                corners,
                ids,
            )

        cv2.imshow(
            "Detected ArUco Markers",
            cv2.resize(out, window_size),
        )

        if verbose:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_charuco_corners(
        self,
        image,
        image_name=None,
        window_size=(480, 480),
        verbose=True,
    ):
        out = image.copy()

        ids, corners = self.detect_charuco_corners(
            out,
            image_name=image_name,
            verbose=verbose,
        )

        if ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                out,
                corners,
                ids,
            )

        cv2.imshow(
            "Detected ChArUco Corners",
            cv2.resize(out, window_size),
        )

        if verbose:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_camera_parameters(self, file_path):
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True,
        )

        with open(file_path, "w") as f:
            json.dump(
                {
                    "K": self.K.tolist(),
                    "D": self.D.tolist(),
                },
                f,
                indent=4,
            )

    def load_camera_parameters(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        self.K = np.array(
            data["K"],
            dtype=np.float64,
        )

        self.D = np.array(
            data["D"],
            dtype=np.float64,
        )

    def _intrinsics_path(self, filename):
        return os.path.abspath(
            os.path.join(
                self.cur_dir,
                "../../data/calibration/camera_intrinsics",
                filename,
            )
        )


class FisheyeCalibrator(CharucoCalibrator):
    def calibrate(
        self,
        grayscale=True,
        calibration_filename="fisheye_calibration.json",
        window_size=(480, 480),
        verbose=False,
    ):
        object_points = []
        image_points = []

        board_points = (
            self.board.getChessboardCorners()
        )

        image_size = None
        count = 0

        for image_name, image in self.calibration_images():
            h, w = image.shape[:2]

            if image_size is None:
                image_size = (w, h)

            if image_size != (w, h):
                print(
                    f"Skipping {image_name}: different resolution"
                )
                continue

            ids, corners = self.detect_charuco_corners(
                image,
                image_name,
                grayscale,
                verbose,
            )

            if ids is None or len(ids) <= 4:
                print(
                    f"Could not interpolate corners for {image_name}"
                )
                continue

            ids_flat = (
                np.asarray(ids)
                .reshape(-1)
                .astype(np.int32)
            )

            object_points.append(
                board_points[ids_flat]
                .astype(np.float64)
                .reshape(-1, 1, 3)
            )

            image_points.append(
                np.asarray(
                    corners,
                    dtype=np.float64,
                ).reshape(-1, 1, 2)
            )

            count += 1

        if count == 0:
            print("No valid ChArUco images found.")
            return

        self.K = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        self.D = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        flags = (
            fisheye_flag(
                "CALIB_RECOMPUTE_EXTRINSIC"
            )
            | fisheye_flag(
                "CALIB_FIX_SKEW"
            )
        )

        rms, self.K, self.D, _, _ = (
            cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                self.K,
                self.D,
                flags=flags,
            )
        )

        print(
            f"Number of images used for calibration: {count}"
        )

        print(
            f"RMS reprojection error: {rms:.6f} px"
        )

        print("K:\n", self.K)

        print(
            "Distortion coefficients:\n",
            self.D,
        )

        output_path = self._intrinsics_path(
            calibration_filename
        )

        self.save_camera_parameters(
            output_path
        )

        print(
            f"Camera calibration data saved to {output_path}"
        )

    def undistort_image(
        self,
        image,
        image_name=None,
        calibration_filename="fisheye_calibration.json",
        balance=0.6,
        show_image=True,
        save_image=True,
        output_path=None,
        window_size=(480, 480),
    ):
        self.load_camera_parameters(
            self._intrinsics_path(
                calibration_filename
            )
        )

        h, w = image.shape[:2]

        new_K = (
            cv2.fisheye
            .estimateNewCameraMatrixForUndistortRectify(
                self.K,
                self.D,
                (w, h),
                np.eye(3),
                balance=balance,
            )
        )

        map1, map2 = (
            cv2.fisheye
            .initUndistortRectifyMap(
                self.K,
                self.D,
                np.eye(3),
                new_K,
                (w, h),
                cv2.CV_16SC2,
            )
        )

        undistorted = cv2.remap(
            image,
            map1,
            map2,
            cv2.INTER_LINEAR,
        )

        if show_image:
            cv2.imshow(
                "Undistorted Image",
                cv2.resize(
                    undistorted,
                    window_size,
                ),
            )

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_image:
            if output_path is None:
                name = (
                    image_name
                    or "undistorted.png"
                )

                output_path = os.path.abspath(
                    os.path.join(
                        self.cur_dir,
                        "../../data/undistorted_images",
                        name,
                    )
                )

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True,
            )

            cv2.imwrite(
                output_path,
                undistorted,
            )

        return undistorted

    def undistort_image_proper(
        self,
        image,
        balance=0.6,
    ):
        h, w = image.shape[:2]

        new_K = (
            cv2.fisheye
            .estimateNewCameraMatrixForUndistortRectify(
                self.K,
                self.D,
                (w, h),
                np.eye(3),
                balance=balance,
            )
        )

        map1, map2 = (
            cv2.fisheye
            .initUndistortRectifyMap(
                self.K,
                self.D,
                np.eye(3),
                new_K,
                (w, h),
                cv2.CV_16SC2,
            )
        )

        return cv2.remap(
            image,
            map1,
            map2,
            cv2.INTER_LINEAR,
        )

    def export_camera_params_colmap(
        self,
        calibration_path=None,
    ):
        if calibration_path is None:
            calibration_path = (
                self._intrinsics_path(
                    "fisheye_calibration.json"
                )
            )

        self.load_camera_parameters(
            calibration_path
        )

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        k1, k2, k3, k4 = (
            self.D.reshape(-1)
        )

        print("Camera parameters for COLMAP:")

        print(
            f"OPENCV_FISHEYE: "
            f"{fx}, {fy}, "
            f"{cx}, {cy}, "
            f"{k1}, {k2}, "
            f"{k3}, {k4}"
        )


class PinholeCalibrator(CharucoCalibrator):
    def calibrate(
        self,
        grayscale=True,
        calibration_filename="pinhole_calibration.json",
        window_size=(480, 480),
        verbose=False,
    ):
        all_corners = []
        all_ids = []

        image_size = None

        for image_name, image in self.calibration_images():
            h, w = image.shape[:2]

            if image_size is None:
                image_size = (w, h)

            if image_size != (w, h):
                print(
                    f"Skipping {image_name}: different resolution"
                )
                continue

            ids, corners = self.detect_charuco_corners(
                image,
                image_name,
                grayscale,
                verbose,
            )

            if ids is None or len(ids) <= 4:
                print(
                    f"Could not interpolate corners for {image_name}"
                )
                continue

            all_ids.append(ids)
            all_corners.append(corners)

        if not all_corners:
            print("No valid ChArUco images found.")
            return

        if hasattr(
            cv2.aruco,
            "calibrateCameraCharuco",
        ):
            rms, self.K, self.D, _, _ = (
                cv2.aruco.calibrateCameraCharuco(
                    all_corners,
                    all_ids,
                    self.board,
                    image_size,
                    None,
                    None,
                )
            )

        else:
            board_points = (
                self.board.getChessboardCorners()
            )

            object_points = []
            image_points = []

            for corners, ids in zip(
                all_corners,
                all_ids,
            ):
                ids_flat = (
                    np.asarray(ids)
                    .reshape(-1)
                    .astype(np.int32)
                )

                object_points.append(
                    board_points[ids_flat]
                    .astype(np.float32)
                    .reshape(-1, 1, 3)
                )

                image_points.append(
                    np.asarray(
                        corners,
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                )

            rms, self.K, self.D, _, _ = (
                cv2.calibrateCamera(
                    object_points,
                    image_points,
                    image_size,
                    None,
                    None,
                )
            )

        print(
            f"Number of images used for calibration: "
            f"{len(all_corners)}"
        )

        print(
            f"RMS reprojection error: {rms:.6f} px"
        )

        print("K:\n", self.K)

        print(
            "Distortion coefficients:\n",
            self.D,
        )

        output_path = self._intrinsics_path(
            calibration_filename
        )

        self.save_camera_parameters(
            output_path
        )

        print(
            f"Camera calibration data saved to {output_path}"
        )

    def undistort_image(
        self,
        image,
        image_name=None,
        calibration_filename="pinhole_calibration.json",
        balance=1.0,
        show_image=True,
        save_image=True,
        output_path=None,
        window_size=(480, 480),
    ):
        self.load_camera_parameters(
            self._intrinsics_path(
                calibration_filename
            )
        )

        h, w = image.shape[:2]

        new_K, _ = (
            cv2.getOptimalNewCameraMatrix(
                self.K,
                self.D,
                (w, h),
                balance,
                (w, h),
            )
        )

        undistorted = cv2.undistort(
            image,
            self.K,
            self.D,
            None,
            new_K,
        )

        if show_image:
            cv2.imshow(
                "Undistorted Image",
                cv2.resize(
                    undistorted,
                    window_size,
                ),
            )

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_image:
            if output_path is None:
                name = (
                    image_name
                    or "undistorted.png"
                )

                output_path = os.path.abspath(
                    os.path.join(
                        self.cur_dir,
                        "../../data/undistorted_images",
                        name,
                    )
                )

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True,
            )

            cv2.imwrite(
                output_path,
                undistorted,
            )

        return undistorted

    def export_camera_params_colmap(
        self,
        calibration_path=None,
    ):
        if calibration_path is None:
            calibration_path = (
                self._intrinsics_path(
                    "pinhole_calibration.json"
                )
            )

        self.load_camera_parameters(
            calibration_path
        )

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        d = self.D.reshape(-1)

        k1 = d[0] if len(d) > 0 else 0.0
        k2 = d[1] if len(d) > 1 else 0.0
        p1 = d[2] if len(d) > 2 else 0.0
        p2 = d[3] if len(d) > 3 else 0.0

        print("Camera parameters for COLMAP:")

        print(
            f"OPENCV: "
            f"{fx}, {fy}, "
            f"{cx}, {cy}, "
            f"{k1}, {k2}, "
            f"{p1}, {p2}"
        )
