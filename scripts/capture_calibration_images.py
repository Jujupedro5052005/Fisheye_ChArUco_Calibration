#!/usr/bin/env python3

import sys
import argparse
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import FisheyeCalibrator


# ============================================================
# CONFIGURATION
# ============================================================

ARUCO_DICT = cv2.aruco.DICT_5X5_100

SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.028

DEFAULT_DEVICE = "/dev/video5"

WIDTH = 1920
HEIGHT = 1536
FPS = 30

CALIBRATION_DIR = REPO_ROOT / "data/calibration/images"
RAW_DIR = REPO_ROOT / "data/raw_images"

MIN_CHARUCO_CORNERS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture ChArUco calibration images."
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Camera device (default: {DEFAULT_DEVICE})",
    )

    return parser.parse_args()


def create_calibrator():
    return FisheyeCalibrator(
        ARUCO_DICT,
        SQUARES_VERTICALLY,
        SQUARES_HORIZONTALLY,
        SQUARE_LENGTH,
        MARKER_LENGTH,
    )


def open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"UYVY"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {device}")

    return cap


def filename(prefix, index, corners):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    return (
        f"{prefix}_{index:04d}_"
        f"{timestamp}_"
        f"{corners:02d}corners.png"
    )


def draw_status(frame, corners, calibration_count, raw_count):
    valid = corners >= MIN_CHARUCO_CORNERS

    lines = [
        f"Charuco corners: {corners}",
        f"Calibration images: {calibration_count}",
        f"Raw images: {raw_count}",
        "",
        "SPACE / S : save calibration image",
        "F         : save raw image",
        "Q / ESC   : quit",
    ]

    color = (0, 255, 0) if valid else (0, 0, 255)

    y = 40

    for line in lines:
        cv2.putText(
            frame,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color if y == 40 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 35


def main():
    args = parse_args()

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    calibrator = create_calibrator()
    cap = open_camera(args.device)

    calibration_count = 0
    raw_count = 0

    print()
    print("=" * 60)
    print("CHARUCO IMAGE CAPTURE")
    print("=" * 60)
    print(f"Device:      {args.device}")
    print(f"Resolution:  {WIDTH} x {HEIGHT}")
    print(f"FPS:         {FPS}")
    print(f"Calibration: {CALIBRATION_DIR}")
    print(f"Raw images:  {RAW_DIR}")
    print()
    print("SPACE / S -> save valid calibration frame")
    print("F         -> force-save raw frame")
    print("Q / ESC   -> quit")
    print()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Failed to read camera frame.")
                continue

            ids, corners = calibrator.detect_charuco_corners(
                frame,
                verbose=False,
            )

            num_corners = 0 if ids is None else len(ids)
            valid = num_corners >= MIN_CHARUCO_CORNERS

            display = frame.copy()

            if ids is not None and corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    display,
                    corners,
                    ids,
                )

            draw_status(
                display,
                num_corners,
                calibration_count,
                raw_count,
            )

            preview_width = 1280
            scale = preview_width / display.shape[1]

            preview = cv2.resize(
                display,
                (
                    preview_width,
                    int(display.shape[0] * scale),
                ),
            )

            cv2.imshow("ChArUco Capture", preview)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key in (ord("s"), ord(" ")):
                if not valid:
                    print(
                        f"Not saved: only {num_corners} "
                        f"ChArUco corners detected."
                    )
                    continue

                name = filename(
                    "calib",
                    calibration_count,
                    num_corners,
                )

                path = CALIBRATION_DIR / name
                cv2.imwrite(str(path), frame)

                calibration_count += 1

                print(f"Saved calibration image: {path}")

            elif key == ord("f"):
                name = filename(
                    "raw",
                    raw_count,
                    num_corners,
                )

                path = RAW_DIR / name
                cv2.imwrite(str(path), frame)

                raw_count += 1

                print(f"Saved raw image: {path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print(f"Calibration images captured: {calibration_count}")
    print(f"Raw images captured:         {raw_count}")


if __name__ == "__main__":
    main()