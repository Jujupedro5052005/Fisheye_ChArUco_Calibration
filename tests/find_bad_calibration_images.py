#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
IMAGE_DIR = ROOT / "data/calibration/images"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import FisheyeCalibrator


ARUCO_DICT = cv2.aruco.DICT_5X5_100

SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5

SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.028


def fisheye_flag(name):
    if hasattr(cv2.fisheye, name):
        return getattr(cv2.fisheye, name)

    if hasattr(cv2, name):
        return getattr(cv2, name)

    raise AttributeError(
        f"OpenCV fisheye flag not found: {name}"
    )


def run_calibration(samples, image_size):
    if len(samples) < 3:
        return False, None, None, None

    K = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    D = np.zeros(
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

    object_points = [
        sample["obj"]
        for sample in samples
    ]

    image_points = [
        sample["img"]
        for sample in samples
    ]

    try:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            K,
            D,
            flags=flags,
        )

        return True, rms, K, D

    except cv2.error:
        return False, None, None, None


def load_samples():
    calibrator = FisheyeCalibrator(
        ARUCO_DICT,
        SQUARES_VERTICALLY,
        SQUARES_HORIZONTALLY,
        SQUARE_LENGTH,
        MARKER_LENGTH,
        calibration_images_dir=str(IMAGE_DIR),
    )

    board_points = (
        calibrator.board
        .getChessboardCorners()
    )

    samples = []
    image_size = None

    print()
    print("Loading calibration observations...")
    print()

    for name, image in calibrator.calibration_images():
        h, w = image.shape[:2]

        if image_size is None:
            image_size = (w, h)

        elif image_size != (w, h):
            print(
                f"SKIP {name}: "
                f"resolution {w}x{h} "
                f"does not match "
                f"{image_size[0]}x{image_size[1]}"
            )
            continue

        ids, corners = calibrator.detect_charuco_corners(
            image=image,
            image_name=name,
            verbose=False,
        )

        if ids is None or corners is None:
            print(
                f"SKIP {name}: "
                "no ChArUco detection"
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

        unique_ids = len(
            np.unique(ids_flat)
        )

        finite = (
            np.isfinite(obj).all()
            and np.isfinite(img).all()
        )

        pts = img.reshape(-1, 2)

        hull = cv2.convexHull(
            pts.astype(np.float32)
        )

        hull_area = cv2.contourArea(
            hull
        )

        x_min = float(
            np.min(pts[:, 0])
        )

        x_max = float(
            np.max(pts[:, 0])
        )

        y_min = float(
            np.min(pts[:, 1])
        )

        y_max = float(
            np.max(pts[:, 1])
        )

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        image_area = w * h

        coverage = (
            100.0
            * hull_area
            / image_area
        )

        print(
            f"{name}: "
            f"{len(ids_flat):2d} corners | "
            f"{unique_ids:2d} unique | "
            f"hull={hull_area:10.1f} px² | "
            f"coverage={coverage:6.3f}% | "
            f"bbox={bbox_width:7.1f}x"
            f"{bbox_height:7.1f} | "
            f"finite={finite}"
        )

        samples.append(
            {
                "name": name,
                "obj": obj,
                "img": img,
                "ids": ids_flat,
                "corners": len(ids_flat),
                "hull_area": hull_area,
                "coverage": coverage,
            }
        )

    return samples, image_size


def test_full_dataset(
    samples,
    image_size,
):
    print()
    print("=" * 70)
    print("FULL DATASET TEST")
    print("=" * 70)

    ok, rms, K, D = run_calibration(
        samples,
        image_size,
    )

    if ok:
        print()
        print("Full dataset: OK")
        print(
            f"RMS: {rms:.6f} px"
        )
        print()
        print("K:")
        print(K)
        print()
        print("D:")
        print(D)

    else:
        print()
        print("Full dataset: FAILED")

    return ok


def leave_one_out_test(
    samples,
    image_size,
):
    print()
    print("=" * 70)
    print("LEAVE-ONE-OUT TEST")
    print("=" * 70)
    print()

    fixes = []

    for i, sample in enumerate(samples):
        subset = (
            samples[:i]
            + samples[i + 1:]
        )

        ok, rms, _, _ = run_calibration(
            subset,
            image_size,
        )

        if ok:
            print(
                f"FIXES DATASET -> "
                f"{sample['name']} "
                f"| corners={sample['corners']} "
                f"| coverage="
                f"{sample['coverage']:.3f}% "
                f"| RMS={rms:.6f}"
            )

            fixes.append(
                sample
            )

        else:
            print(
                f"still fails     "
                f"{sample['name']} "
                f"| corners="
                f"{sample['corners']} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

    return fixes


def incremental_test(
    samples,
    image_size,
):
    print()
    print("=" * 70)
    print("INCREMENTAL TEST")
    print("=" * 70)
    print()

    good = []
    rejected = []

    for sample in samples:
        candidate = (
            good
            + [sample]
        )

        if len(candidate) < 3:
            good.append(sample)

            print(
                f"SEED   "
                f"{sample['name']} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

            continue

        ok, rms, _, _ = run_calibration(
            candidate,
            image_size,
        )

        if ok:
            good.append(sample)

            print(
                f"KEEP   "
                f"{sample['name']} "
                f"| n={len(good):2d} "
                f"| RMS={rms:.6f} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

        else:
            rejected.append(sample)

            print(
                f"REJECT "
                f"{sample['name']} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

    return good, rejected


def reverse_incremental_test(
    samples,
    image_size,
):
    print()
    print("=" * 70)
    print("REVERSE INCREMENTAL TEST")
    print("=" * 70)
    print()

    ordered = sorted(
        samples,
        key=lambda sample:
        sample["coverage"],
        reverse=True,
    )

    good = []
    rejected = []

    for sample in ordered:
        candidate = (
            good
            + [sample]
        )

        if len(candidate) < 3:
            good.append(sample)

            print(
                f"SEED   "
                f"{sample['name']} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

            continue

        ok, rms, _, _ = run_calibration(
            candidate,
            image_size,
        )

        if ok:
            good.append(sample)

            print(
                f"KEEP   "
                f"{sample['name']} "
                f"| n={len(good):2d} "
                f"| RMS={rms:.6f} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

        else:
            rejected.append(sample)

            print(
                f"REJECT "
                f"{sample['name']} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

    return good, rejected


def print_final_subset(
    good,
    rejected,
    image_size,
):
    print()
    print("=" * 70)
    print("FINAL DIAGNOSTIC RESULT")
    print("=" * 70)

    print()
    print(
        f"Accepted images: {len(good)}"
    )

    print(
        f"Rejected images: {len(rejected)}"
    )

    if rejected:
        print()
        print("Rejected:")

        for sample in rejected:
            print(
                f"  {sample['name']} "
                f"| corners="
                f"{sample['corners']} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

    if len(good) < 3:
        print()
        print(
            "Not enough valid observations "
            "to calibrate."
        )
        return

    ok, rms, K, D = run_calibration(
        good,
        image_size,
    )

    print()

    if not ok:
        print(
            "Final subset still fails."
        )
        return

    print(
        "Final subset calibrates successfully."
    )

    print(
        f"RMS: {rms:.6f} px"
    )

    print()
    print("K:")
    print(K)

    print()
    print("D:")
    print(D)


def main():
    print()
    print("=" * 70)
    print("FISHEYE CALIBRATION DATASET DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        f"OpenCV: {cv2.__version__}"
    )

    print(
        f"Images: {IMAGE_DIR}"
    )

    samples, image_size = (
        load_samples()
    )

    print()
    print(
        f"Total observations: "
        f"{len(samples)}"
    )

    if image_size is None:
        raise RuntimeError(
            "No calibration images found."
        )

    if len(samples) < 3:
        raise RuntimeError(
            "At least 3 valid observations "
            "are required."
        )

    if test_full_dataset(
        samples,
        image_size,
    ):
        return

    fixes = leave_one_out_test(
        samples,
        image_size,
    )

    if fixes:
        print()
        print("=" * 70)
        print("LIKELY PROBLEMATIC IMAGE(S)")
        print("=" * 70)
        print()

        for sample in fixes:
            print(
                f"{sample['name']} "
                f"| corners="
                f"{sample['corners']} "
                f"| hull="
                f"{sample['hull_area']:.1f} "
                f"| coverage="
                f"{sample['coverage']:.3f}%"
            )

        return

    print()
    print(
        "No single image removal fixed "
        "the complete dataset."
    )

    good, rejected = (
        incremental_test(
            samples,
            image_size,
        )
    )

    print_final_subset(
        good,
        rejected,
        image_size,
    )

    print()
    print(
        "Running second incremental test "
        "starting with the largest board views..."
    )

    good, rejected = (
        reverse_incremental_test(
            samples,
            image_size,
        )
    )

    print_final_subset(
        good,
        rejected,
        image_size,
    )


if __name__ == "__main__":
    main()