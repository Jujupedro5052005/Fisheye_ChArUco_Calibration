#!/usr/bin/env python3

import json
import os
from itertools import combinations

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

        if hasattr(cv2.aruco, "CharucoBoard"):
            self.board = cv2.aruco.CharucoBoard(
                (squares_vertically, squares_horizontally),
                square_length,
                marker_length,
                self.aruco_dict,
            )
        else:
            self.board = cv2.aruco.CharucoBoard_create(
                squares_vertically,
                squares_horizontally,
                square_length,
                marker_length,
                self.aruco_dict,
            )

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.params = cv2.aruco.DetectorParameters()
        else:
            self.params = cv2.aruco.DetectorParameters_create()

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

            image = cv2.imread(os.path.join(folder_path, name))

            if image is not None:
                yield name, image

    def calibration_images(self):
        yield from self.image_generator(self.calibration_image_dir)

    def raw_images(self):
        yield from self.image_generator(self.raw_image_dir)

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

            marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

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
            marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.params,
            )

            if marker_ids is not None and len(marker_ids):
                try:
                    result = cv2.aruco.refineDetectedMarkers(
                        gray,
                        self.board,
                        marker_corners,
                        marker_ids,
                        rejected,
                        parameters=self.params,
                    )

                    marker_corners, marker_ids = result[:2]

                except cv2.error:
                    pass

        if marker_ids is None or len(marker_ids) == 0:
            if verbose:
                print(f"No ArUco markers detected in {image_name}")

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
            detector = cv2.aruco.CharucoDetector(self.board)

            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) == 0:
            return None, None

        if verbose:
            print(
                f"{len(charuco_ids)} ChArUco corners detected "
                f"in {image_name}"
            )

        return charuco_ids, charuco_corners

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

        self.K = np.array(data["K"], dtype=np.float64)
        self.D = np.array(data["D"], dtype=np.float64)

    def _intrinsics_path(self, filename):
        return os.path.abspath(
            os.path.join(
                self.cur_dir,
                "../../data/calibration/camera_intrinsics",
                filename,
            )
        )


class FisheyeCalibrator(CharucoCalibrator):
    def _fit(self, samples, image_size):
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        flags = (
            fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC")
            | fisheye_flag("CALIB_FIX_SKEW")
        )

        return cv2.fisheye.calibrate(
            [sample["obj"] for sample in samples],
            [sample["img"] for sample in samples],
            image_size,
            K,
            D,
            flags=flags,
        )

    def _try_fit(self, samples, image_size):
        if len(samples) < 3:
            return None

        try:
            return self._fit(samples, image_size)

        except cv2.error:
            return None

    def _find_stable_subset(self, samples, image_size):
        print()
        print(
            "Fisheye calibration failed with the complete dataset."
        )
        print(
            "Searching automatically for unstable observations..."
        )

        ordered = sorted(
            samples,
            key=lambda sample: sample["coverage"],
            reverse=True,
        )

        pool = ordered[:min(10, len(ordered))]

        best_seed = None
        best_result = None

        for seed_size in (5, 4, 3):
            if len(pool) < seed_size:
                continue

            for indices in combinations(
                range(len(pool)),
                seed_size,
            ):
                candidate = [
                    pool[i]
                    for i in indices
                ]

                result = self._try_fit(
                    candidate,
                    image_size,
                )

                if result is None:
                    continue

                if (
                    best_result is None
                    or result[0] < best_result[0]
                ):
                    best_seed = candidate
                    best_result = result

            if best_seed is not None:
                break

        if best_seed is None:
            return None, [], samples

        accepted = list(best_seed)

        accepted_names = {
            sample["name"]
            for sample in accepted
        }

        rejected = []

        for sample in ordered:
            if sample["name"] in accepted_names:
                continue

            candidate = accepted + [sample]

            result = self._try_fit(
                candidate,
                image_size,
            )

            if result is not None:
                accepted.append(sample)
                accepted_names.add(sample["name"])
            else:
                rejected.append(sample)

        final_result = self._try_fit(
            accepted,
            image_size,
        )

        return final_result, accepted, rejected

    def calibrate(
        self,
        grayscale=True,
        calibration_filename="fisheye_calibration.json",
        window_size=(480, 480),
        verbose=False,
    ):
        board_points = self.board.getChessboardCorners()

        samples = []
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
                image=image,
                image_name=image_name,
                grayscale=grayscale,
                verbose=verbose,
            )

            if ids is None or len(ids) <= 4:
                print(
                    f"Skipping {image_name}: "
                    "insufficient ChArUco corners"
                )
                continue

            ids_flat = (
                np.asarray(ids)
                .reshape(-1)
                .astype(np.int32)
            )

            obj = (
                board_points[ids_flat]
                .astype(np.float64)
                .reshape(-1, 1, 3)
            )

            img = (
                np.asarray(
                    corners,
                    dtype=np.float64,
                )
                .reshape(-1, 1, 2)
            )

            points = img.reshape(-1, 2)

            hull = cv2.convexHull(
                points.astype(np.float32)
            )

            hull_area = cv2.contourArea(hull)

            coverage = (
                hull_area
                / float(w * h)
            )

            samples.append(
                {
                    "name": image_name,
                    "obj": obj,
                    "img": img,
                    "coverage": coverage,
                    "corners": len(ids_flat),
                }
            )

        if len(samples) < 3:
            print(
                "Not enough valid images for fisheye calibration."
            )
            return False

        print(
            f"Candidate images for calibration: {len(samples)}"
        )

        result = self._try_fit(
            samples,
            image_size,
        )

        accepted = samples
        rejected = []

        if result is None:
            result, accepted, rejected = (
                self._find_stable_subset(
                    samples,
                    image_size,
                )
            )

        if result is None:
            print()
            print(
                "Fisheye calibration could not be completed."
            )
            print(
                "Pinhole calibration can still be attempted."
            )

            return False

        rms, self.K, self.D, _, _ = result

        print()
        print(
            f"Number of images used for calibration: "
            f"{len(accepted)}"
        )

        if rejected:
            print(
                f"Automatically rejected unstable views: "
                f"{len(rejected)}"
            )

            for sample in rejected:
                print(
                    f"  - {sample['name']} "
                    f"({sample['corners']} corners, "
                    f"{sample['coverage'] * 100:.3f}% coverage)"
                )

        print()
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
            f"Camera calibration data saved to "
            f"{output_path}"
        )

        return True

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
                image=image,
                image_name=image_name,
                grayscale=grayscale,
                verbose=verbose,
            )

            if ids is None or len(ids) <= 4:
                print(
                    f"Skipping {image_name}: "
                    "insufficient ChArUco corners"
                )
                continue

            all_ids.append(ids)
            all_corners.append(corners)

        if not all_corners:
            print(
                "No valid ChArUco images found."
            )
            return False

        try:
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
                    self.board
                    .getChessboardCorners()
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
                        )
                        .reshape(-1, 1, 2)
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

        except cv2.error as exc:
            print()
            print(
                "Pinhole calibration failed."
            )
            print(exc)

            return False

        print()
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
            f"Camera calibration data saved to "
            f"{output_path}"
        )

        return True

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

        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K,
            self.D,
            (w, h),
            balance,
            (w, h),
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

        print(
            "Camera parameters for COLMAP:"
        )

        print(
            f"OPENCV: "
            f"{fx}, {fy}, "
            f"{cx}, {cy}, "
            f"{k1}, {k2}, "
            f"{p1}, {p2}"
        )