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

RUN_FISHEYE = True
RUN_PINHOLE = True


# ============================================================
# PATHS
# ============================================================

CALIBRATION_IMAGES_DIR = (
    REPO_ROOT
    / "data"
    / "calibration"
    / "images"
)

RAW_IMAGES_DIR = (
    REPO_ROOT
    / "data"
    / "raw_images"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "calibration"
    / "camera_intrinsics"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def count_images(folder):
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_calibrator(cls):
    return cls(
        ARUCO_DICT,
        SQUARES_VERTICALLY,
        SQUARES_HORIZONTALLY,
        SQUARE_LENGTH,
        MARKER_LENGTH,
        calibration_images_dir=str(
            CALIBRATION_IMAGES_DIR
        ),
        raw_images_dir=str(
            RAW_IMAGES_DIR
        ),
    )


def main():
    CALIBRATION_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_count = count_images(
        CALIBRATION_IMAGES_DIR
    )

    print()
    print("=" * 70)
    print("CAMERA INTRINSIC CALIBRATION")
    print("=" * 70)

    print()
    print("Repository:")
    print(f"  {REPO_ROOT}")

    print()
    print("Calibration dataset:")
    print(f"  {CALIBRATION_IMAGES_DIR}")

    print()
    print(
        f"Calibration images found: "
        f"{image_count}"
    )

    print()
    print("ChArUco board configuration:")
    print(
        "  Dictionary        : "
        "DICT_5X5_100"
    )
    print(
        f"  Board size        : "
        f"{SQUARES_VERTICALLY} x "
        f"{SQUARES_HORIZONTALLY} squares"
    )
    print(
        f"  Square length     : "
        f"{SQUARE_LENGTH * 1000:.1f} mm"
    )
    print(
        f"  Marker length     : "
        f"{MARKER_LENGTH * 1000:.1f} mm"
    )

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    if image_count == 0:
        print()
        print(
            "No calibration images found."
        )
        print(
            "Capture images first with:"
        )
        print(
            "  python "
            "scripts/capture_calibration_images.py"
        )
        return

    fisheye_success = False
    pinhole_success = False

    # ========================================================
    # FISHEYE
    # ========================================================

    if RUN_FISHEYE:
        print()
        print()
        print("=" * 70)
        print("FISHEYE CALIBRATION")
        print("=" * 70)
        print()

        fisheye = create_calibrator(
            FisheyeCalibrator
        )

        try:
            fisheye_success = (
                fisheye.calibrate()
            )

        except Exception as exc:
            print()
            print(
                "Unexpected fisheye calibration error."
            )
            print(exc)
            fisheye_success = False

        if fisheye_success:
            print()
            print(
                "Fisheye parameters in COLMAP format:"
            )

            fisheye.export_camera_params_colmap()

        else:
            print()
            print(
                "Fisheye calibration was not completed."
            )
            print(
                "Continuing with pinhole calibration..."
            )

    # ========================================================
    # PINHOLE
    # ========================================================

    if RUN_PINHOLE:
        print()
        print()
        print("=" * 70)
        print("PINHOLE CALIBRATION")
        print("=" * 70)
        print()

        pinhole = create_calibrator(
            PinholeCalibrator
        )

        try:
            pinhole_success = (
                pinhole.calibrate()
            )

        except Exception as exc:
            print()
            print(
                "Unexpected pinhole calibration error."
            )
            print(exc)
            pinhole_success = False

        if pinhole_success:
            print()
            print(
                "Pinhole parameters in COLMAP format:"
            )

            pinhole.export_camera_params_colmap()

        else:
            print()
            print(
                "Pinhole calibration was not completed."
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    if RUN_FISHEYE:
        print(
            "Fisheye : "
            + (
                "SUCCESS"
                if fisheye_success
                else "FAILED"
            )
        )

    if RUN_PINHOLE:
        print(
            "Pinhole : "
            + (
                "SUCCESS"
                if pinhole_success
                else "FAILED"
            )
        )

    print()

    if fisheye_success or pinhole_success:
        print(
            "Results saved under:"
        )
        print(
            f"  {OUTPUT_DIR}"
        )

        if fisheye_success:
            print(
                "  - fisheye_calibration.json"
            )

        if pinhole_success:
            print(
                "  - pinhole_calibration.json"
            )

    else:
        print(
            "No calibration model completed successfully."
        )

    print()


if __name__ == "__main__":
    main()