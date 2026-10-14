#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import FisheyeCalibrator, PinholeCalibrator


# ============================================================
# CONFIGURATION
# ============================================================

ARUCO_DICT = cv2.aruco.DICT_5X5_100

SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5

SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.028

MODEL = "fisheye"      # "fisheye" or "pinhole"
BALANCE = 0.6

SHOW_IMAGES = True
SAVE_IMAGES = True


# ============================================================
# PATHS
# ============================================================

RAW_IMAGES_DIR = REPO_ROOT / "data/raw_images"

OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "undistorted_images"
)

INTRINSICS_DIR = (
    REPO_ROOT
    / "data"
    / "calibration"
    / "camera_intrinsics"
)


def create_calibrator():
    cls = (
        FisheyeCalibrator
        if MODEL == "fisheye"
        else PinholeCalibrator
    )

    return cls(
        ARUCO_DICT,
        SQUARES_VERTICALLY,
        SQUARES_HORIZONTALLY,
        SQUARE_LENGTH,
        MARKER_LENGTH,
        raw_images_dir=str(RAW_IMAGES_DIR),
    )


def main():
    RAW_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MODEL not in {"fisheye", "pinhole"}:
        raise ValueError(
            'MODEL must be "fisheye" or "pinhole".'
        )

    calibration_filename = (
        "fisheye_calibration.json"
        if MODEL == "fisheye"
        else "pinhole_calibration.json"
    )

    calibration_path = (
        INTRINSICS_DIR
        / calibration_filename
    )

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n"
            f"{calibration_path}\n\n"
            "Run first:\n"
            "  python scripts/run_calibration.py"
        )

    calibrator = create_calibrator()

    images = list(
        calibrator.raw_images()
    )

    print()
    print("=" * 60)
    print("IMAGE UNDISTORTION")
    print("=" * 60)

    print(f"Model:       {MODEL}")
    print(f"Calibration: {calibration_path}")
    print(f"Input:       {RAW_IMAGES_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Images:      {len(images)}")

    if MODEL == "fisheye":
        print(f"Balance:     {BALANCE}")

    if not images:
        raise RuntimeError(
            f"No images found in:\n{RAW_IMAGES_DIR}"
        )

    for i, (name, image) in enumerate(
        images,
        start=1,
    ):
        print(
            f"[{i}/{len(images)}] {name}"
        )

        output_path = (
            OUTPUT_DIR
            / name
        )

        calibrator.undistort_image(
            image=image,
            image_name=name,
            calibration_filename=calibration_filename,
            balance=BALANCE,
            show_image=SHOW_IMAGES,
            save_image=SAVE_IMAGES,
            output_path=str(output_path),
        )

    print()
    print("=" * 60)
    print("UNDISTORTION FINISHED")
    print("=" * 60)

    if SAVE_IMAGES:
        print(f"Results: {OUTPUT_DIR}")

    print()


if __name__ == "__main__":
    main()
