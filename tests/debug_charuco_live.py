import os
import cv2
import numpy as np
from calibration import FisheyeCalibrator

# ==========================================
# CONFIG (same as yours)
# ==========================================
ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.028

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

instaCamFisheye = FisheyeCalibrator(
    ARUCO_DICT,
    SQUARES_VERTICALLY,
    SQUARES_HORIZONTALLY,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    calibration_images_dir=os.path.join(CURRENT_PATH,'../data/calibration/images/'),
    raw_images_dir=os.path.join(CURRENT_PATH,'../data/raw_images/descent_1')
)

# ==========================================
# NEW PIPELINE (your method)
# ==========================================
charuco_detector = cv2.aruco.CharucoDetector(instaCamFisheye.board)

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

WINDOW_NAME = "PIPELINE COMPARISON"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1400, 900)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    display2 = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # REPO PIPELINE
    # ==========================================
    marker_ids, marker_corners = instaCamFisheye.detect_aruco_markers(
        frame.copy(), image_name="live", verbose=False
    )

    repo_charuco_ids = None
    repo_charuco_corners = None
    repo_count = 0

    if marker_ids is not None and len(marker_ids) > 0:
        retval, repo_charuco_corners, repo_charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, instaCamFisheye.board
        )

        if retval and repo_charuco_ids is not None:
            repo_count = len(repo_charuco_ids)

            # draw in RED
            cv2.aruco.drawDetectedCornersCharuco(
                display,
                repo_charuco_corners,
                repo_charuco_ids,
                cornerColor=(0, 0, 255)
            )

    # ==========================================
    # NEW PIPELINE (CharucoDetector)
    # ==========================================
    
    new_charuco_corners, new_charuco_ids, _, _ = charuco_detector.detectBoard(gray)

    new_count = 0

    if new_charuco_ids is not None:
        new_count = len(new_charuco_ids)

        # draw in GREEN
        cv2.aruco.drawDetectedCornersCharuco(
            #display2,
            display,
            new_charuco_corners,
            new_charuco_ids,
            cornerColor=(0, 255, 0)
        )

    # ==========================================
    # DRAW TEXT INFO
    # ==========================================
    cv2.putText(display, f"Repo corners: {repo_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(display, f"New corners: {new_count}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # difference indicator
    diff = new_count - repo_count
    cv2.putText(display, f"Diff: {diff}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, display)
    #cv2.imshow(f"{WINDOW_NAME}2", display2)


    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()