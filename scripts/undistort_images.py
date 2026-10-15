#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import FisheyeCalibrator, PinholeCalibrator


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MODEL = "pinhole"
DEFAULT_BALANCE = 0.6

SHOW_COMPARISON = True
SAVE_UNDISTORTED = True
SAVE_COMPARISON = True

DISPLAY_WIDTH = 1600
DRAW_LABELS = True


# ============================================================
# CAMERA / BOARD CONFIGURATION
# ============================================================

ARUCO_DICT = cv2.aruco.DICT_5X5_100

SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5

SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.028


# ============================================================
# PATHS
# ============================================================

RAW_IMAGES_DIR = (
    REPO_ROOT
    / "data"
    / "raw_images"
)

UNDISTORTED_IMAGES_DIR = (
    REPO_ROOT
    / "data"
    / "undistorted_images"
)

COMPARISON_IMAGES_DIR = (
    REPO_ROOT
    / "data"
    / "undistortion_comparisons"
)

INTRINSICS_DIR = (
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


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Undistort raw camera images and generate "
            "side-by-side original/undistorted comparisons."
        )
    )

    parser.add_argument(
        "--model",
        choices=["pinhole", "fisheye"],
        default=DEFAULT_MODEL,
        help=(
            "Camera model to use. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--balance",
        type=float,
        default=DEFAULT_BALANCE,
        help=(
            "Field-of-view preservation parameter. "
            "Usually between 0.0 and 1.0. "
            f"Default: {DEFAULT_BALANCE}"
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display comparison windows.",
    )

    parser.add_argument(
        "--no-save-comparison",
        action="store_true",
        help="Do not save side-by-side comparison images.",
    )

    parser.add_argument(
        "--no-save-undistorted",
        action="store_true",
        help="Do not save undistorted images.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def create_calibrator(model):
    if model == "fisheye":
        return FisheyeCalibrator(
            ARUCO_DICT,
            SQUARES_VERTICALLY,
            SQUARES_HORIZONTALLY,
            SQUARE_LENGTH,
            MARKER_LENGTH,
            calibration_images_dir=None,
            raw_images_dir=str(RAW_IMAGES_DIR),
        )

    if model == "pinhole":
        return PinholeCalibrator(
            ARUCO_DICT,
            SQUARES_VERTICALLY,
            SQUARES_HORIZONTALLY,
            SQUARE_LENGTH,
            MARKER_LENGTH,
            calibration_images_dir=None,
            raw_images_dir=str(RAW_IMAGES_DIR),
        )

    raise ValueError(
        f"Unsupported model: {model}"
    )


def get_calibration_file(model):
    if model == "fisheye":
        return (
            INTRINSICS_DIR
            / "fisheye_calibration.json"
        )

    return (
        INTRINSICS_DIR
        / "pinhole_calibration.json"
    )


def find_images(folder):
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def undistort_fisheye(
    image,
    K,
    D,
    balance,
):
    h, w = image.shape[:2]

    new_K = (
        cv2.fisheye
        .estimateNewCameraMatrixForUndistortRectify(
            K,
            D,
            (w, h),
            np.eye(3),
            balance=balance,
        )
    )

    map1, map2 = (
        cv2.fisheye
        .initUndistortRectifyMap(
            K,
            D,
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
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def undistort_pinhole(
    image,
    K,
    D,
    balance,
):
    h, w = image.shape[:2]

    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K,
        D,
        (w, h),
        balance,
        (w, h),
    )

    return cv2.undistort(
        image,
        K,
        D,
        None,
        new_K,
    )


def undistort(
    image,
    K,
    D,
    model,
    balance,
):
    if model == "fisheye":
        return undistort_fisheye(
            image,
            K,
            D,
            balance,
        )

    return undistort_pinhole(
        image,
        K,
        D,
        balance,
    )


def add_label(
    image,
    text,
):
    if not DRAW_LABELS:
        return image

    output = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(
        0.8,
        image.shape[1] / 1200.0,
    )

    thickness = max(
        2,
        int(round(font_scale * 2)),
    )

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )

    x = 30
    y = 30 + text_height
    padding = 15

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (
            x - padding,
            y - text_height - padding,
        ),
        (
            x + text_width + padding,
            y + baseline + padding,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        output,
        0.45,
        0,
        output,
    )

    cv2.putText(
        output,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return output


def create_comparison(
    original,
    undistorted,
    model,
):
    if original.shape[:2] != undistorted.shape[:2]:
        undistorted = cv2.resize(
            undistorted,
            (
                original.shape[1],
                original.shape[0],
            ),
        )

    original_labeled = add_label(
        original,
        "ORIGINAL",
    )

    undistorted_labeled = add_label(
        undistorted,
        f"UNDISTORTED - {model.upper()}",
    )

    divider_width = 8

    divider = np.zeros(
        (
            original.shape[0],
            divider_width,
            3,
        ),
        dtype=np.uint8,
    )

    return np.hstack(
        (
            original_labeled,
            divider,
            undistorted_labeled,
        )
    )


def resize_for_display(image):
    if DISPLAY_WIDTH is None:
        return image

    h, w = image.shape[:2]

    if w <= DISPLAY_WIDTH:
        return image

    scale = DISPLAY_WIDTH / float(w)

    display_height = int(
        round(h * scale)
    )

    return cv2.resize(
        image,
        (
            DISPLAY_WIDTH,
            display_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    model = args.model
    balance = args.balance

    show_comparison = not args.no_show
    save_comparison = not args.no_save_comparison
    save_undistorted = not args.no_save_undistorted

    RAW_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    UNDISTORTED_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_file = get_calibration_file(
        model
    )

    print()
    print("=" * 70)
    print("IMAGE UNDISTORTION")
    print("=" * 70)

    print()
    print(f"Model:       {model}")
    print(f"Balance:     {balance}")

    print()
    print("Raw images:")
    print(f"  {RAW_IMAGES_DIR}")

    print()
    print("Calibration:")
    print(f"  {calibration_file}")

    print()
    print("Undistorted output:")
    print(f"  {UNDISTORTED_IMAGES_DIR}")

    print()
    print("Comparison output:")
    print(f"  {COMPARISON_IMAGES_DIR}")

    if not calibration_file.exists():
        raise FileNotFoundError(
            "\nCalibration file not found:\n"
            f"{calibration_file}\n\n"
            "Run first:\n"
            "  python scripts/run_calibration.py"
        )

    images = find_images(
        RAW_IMAGES_DIR
    )

    print()
    print(
        f"Raw images found: {len(images)}"
    )

    if not images:
        raise RuntimeError(
            "\nNo raw images found.\n"
            "Capture one or more raw images first."
        )

    calibrator = create_calibrator(
        model
    )

    calibrator.load_camera_parameters(
        str(calibration_file)
    )

    K = calibrator.K
    D = calibrator.D

    print()
    print("K:")
    print(K)

    print()
    print("D:")
    print(D)

    print()
    print("=" * 70)
    print("PROCESSING")
    print("=" * 70)
    print()

    processed = 0

    for image_path in images:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Skipping unreadable image: "
                f"{image_path.name}"
            )
            continue

        undistorted = undistort(
            image,
            K,
            D,
            model,
            balance,
        )

        comparison = create_comparison(
            image,
            undistorted,
            model,
        )

        if save_undistorted:
            undistorted_path = (
                UNDISTORTED_IMAGES_DIR
                / image_path.name
            )

            cv2.imwrite(
                str(undistorted_path),
                undistorted,
            )

        if save_comparison:
            comparison_path = (
                COMPARISON_IMAGES_DIR
                / (
                    f"{image_path.stem}"
                    f"_{model}_comparison"
                    f"{image_path.suffix}"
                )
            )

            cv2.imwrite(
                str(comparison_path),
                comparison,
            )

        processed += 1

        print(
            f"[{processed:03d}/{len(images):03d}] "
            f"{image_path.name}"
        )

        if show_comparison:
            display = resize_for_display(
                comparison
            )

            cv2.imshow(
                "Original | Undistorted",
                display,
            )

            key = cv2.waitKey(0) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                print()
                print(
                    "Display interrupted by user."
                )
                break

    if show_comparison:
        cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("UNDISTORTION FINISHED")
    print("=" * 70)

    print()
    print(
        f"Processed images: {processed}"
    )

    if save_undistorted:
        print()
        print("Undistorted images:")
        print(
            f"  {UNDISTORTED_IMAGES_DIR}"
        )

    if save_comparison:
        print()
        print("Side-by-side comparisons:")
        print(
            f"  {COMPARISON_IMAGES_DIR}"
        )

    print()


if __name__ == "__main__":
    main()