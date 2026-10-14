#!/usr/bin/env python3

"""
Run camera intrinsic calibration using a ChArUco board.
"""

import sys
from pathlib import Path

import cv2


# ============================================================
# REPOSITORY PATHS
# ============================================================
#
# Repository structure:
#
#   Fisheye_ChArUco_Calibration/
#   ├── scripts/
#   │   └── run_calibration.py
#   └── src/
#       └── calibration/
#
# When this script is executed directly:
#
#   python scripts/run_calibration.py
#
# Python automatically sees scripts/, but not src/.
#
# Therefore we explicitly add:
#
#   repository/src/
#
# to Python's module search path.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# This import now works directly from the repository,
# without requiring `pip install -e .`.
from calibration import (
    FisheyeCalibrator,
    PinholeCalibrator,
)


# ============================================================
# REPOSITORY PATHS
# ============================================================
#
# This file lives in:
#
#   repository/
#       scripts/
#           run_calibration.py
#
# Therefore:
#
#   SCRIPT_DIR = repository/scripts
#   REPO_ROOT  = repository
#
# Using relative repository paths instead of "/home/orin/..."
# makes the project portable to another computer/user.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ============================================================
# CHARUCO / ARUCO CONFIGURATION
# ============================================================
#
# These values MUST describe the REAL printed calibration board.
#
# IMPORTANT:
#
# The OpenCV CharucoBoard constructor ultimately expects:
#
#     (squaresX, squaresY)
#
# In the current calibration class, however, the historical variable names
# are:
#
#     SQUARES_VERTICALLY
#     SQUARES_HORIZONTALLY
#
# and they are passed to CharucoBoard in that same order.
#
# Therefore this setup currently creates:
#
#     CharucoBoard((7, 5), ...)
#
# i.e. 7 squares along X and 5 along Y.
#
# Keep these values consistent with the board actually used.
# ============================================================

ARUCO_DICT = cv2.aruco.DICT_5X5_100

SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5

# Physical side length of each chessboard square [meters]
SQUARE_LENGTH = 0.040

# Physical side length of each internal ArUco marker [meters]
MARKER_LENGTH = 0.028


# ============================================================
# INPUT DIRECTORIES
# ============================================================
#
# CALIBRATION_IMAGES_DIR
#
# Contains images of the ChArUco board used to estimate:
#
#     K = intrinsic camera matrix
#     D = distortion coefficients
#
# This is the IMPORTANT dataset for intrinsic calibration.
#
# Example:
#
#   data/calibration/images/
#
#       image_001.png
#       image_002.png
#       image_003.png
#       ...
#
#
# RAW_IMAGES_DIR
#
# Contains arbitrary normal camera images.
#
# These images are NOT used to calculate K and D by calibrate().
# They are mainly useful for later testing, undistortion and validation.
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


# ============================================================
# OUTPUT DIRECTORY
# ============================================================
#
# The calibrator itself currently writes calibration files to:
#
#   data/calibration/camera_intrinsics/
#
# producing:
#
#   fisheye_calibration.json
#   pinhole_calibration.json
#
# Create the directory here just to guarantee it exists.
# ============================================================

CAMERA_INTRINSICS_DIR = (
    REPO_ROOT
    / "data"
    / "calibration"
    / "camera_intrinsics"
)


# ============================================================
# ENABLE / DISABLE CALIBRATION MODELS
# ============================================================
#
# For your fisheye automotive camera, the fisheye calibration is
# the main result.
#
# Pinhole calibration is kept because it is useful as:
#
#   - a comparison
#   - a debugging/reference model
#
# Set either one to False if you do not want to run it.
# ============================================================

RUN_FISHEYE_CALIBRATION = True
RUN_PINHOLE_CALIBRATION = True


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_paths():
    """
    Check that the expected folders exist before calibration starts.
    """

    CAMERA_INTRINSICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CALIBRATION_IMAGES_DIR.exists():
        raise FileNotFoundError(
            "\nCalibration image directory does not exist:\n"
            f"  {CALIBRATION_IMAGES_DIR}\n\n"
            "Create it and place the ChArUco calibration images there."
        )

    if not RAW_IMAGES_DIR.exists():
        # RAW images are not required for calibration itself.
        #
        # Therefore we create the directory automatically instead
        # of failing.
        RAW_IMAGES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# COUNT CALIBRATION IMAGES
# ============================================================

def count_calibration_images():
    """
    Count image files currently available for calibration.
    """

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
    }

    return sum(
        1
        for path in CALIBRATION_IMAGES_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_extensions
        )
    )


# ============================================================
# CREATE CALIBRATORS
# ============================================================

def create_fisheye_calibrator():
    """
    Create the fisheye calibration object.

    This uses cv2.fisheye.calibrate() internally.
    """

    return FisheyeCalibrator(
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


def create_pinhole_calibrator():
    """
    Create the conventional pinhole calibration object.

    This is mainly useful as a comparison against the fisheye result.
    """

    return PinholeCalibrator(
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


# ============================================================
# MAIN
# ============================================================

def main():

    validate_paths()

    image_count = count_calibration_images()

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
    print("  Dictionary        : DICT_5X5_100")
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
    print(f"  {CAMERA_INTRINSICS_DIR}")

    print()

    if image_count == 0:
        raise RuntimeError(
            "\nNo calibration images were found.\n\n"
            "Capture ChArUco images first using:\n\n"
            "  python scripts/capture_calibration_images.py\n"
        )

    # ========================================================
    # FISHEYE CALIBRATION
    # ========================================================

    if RUN_FISHEYE_CALIBRATION:

        print()
        print("=" * 70)
        print("FISHEYE CALIBRATION")
        print("=" * 70)
        print()

        fisheye_calibrator = (
            create_fisheye_calibrator()
        )

        fisheye_calibrator.calibrate()

        print()
        print(
            "Fisheye parameters "
            "in COLMAP format:"
        )

        fisheye_calibrator.export_camera_params_colmap()

    # ========================================================
    # PINHOLE CALIBRATION
    # ========================================================

    if RUN_PINHOLE_CALIBRATION:

        print()
        print("=" * 70)
        print("PINHOLE CALIBRATION")
        print("=" * 70)
        print()

        pinhole_calibrator = (
            create_pinhole_calibrator()
        )

        pinhole_calibrator.calibrate()

        print()
        print(
            "Pinhole parameters "
            "in COLMAP format:"
        )

        pinhole_calibrator.export_camera_params_colmap()

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("CALIBRATION FINISHED")
    print("=" * 70)

    print()
    print("Results saved under:")
    print(
        f"  {CAMERA_INTRINSICS_DIR}"
    )

    if RUN_FISHEYE_CALIBRATION:
        print(
            "  - fisheye_calibration.json"
        )

    if RUN_PINHOLE_CALIBRATION:
        print(
            "  - pinhole_calibration.json"
        )

    print()


if __name__ == "__main__":
    main()

