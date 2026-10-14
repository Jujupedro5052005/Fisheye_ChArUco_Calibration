import os
import json
import cv2
import numpy as np

# ==========================================
# PATHS
# ==========================================
BASE_PATH = "/home/orin/Documents/PROJETO_ETICO/birds_eye_view/Fisheye_ChArUco_Calibration"

INPUT_FOLDER = os.path.join(BASE_PATH, "data/raw_images/descent_1")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "data/undistorted_images2")
CALIB_FILE = os.path.join(BASE_PATH, "data/calibration/camera_intrinsics/fisheye_calibration.json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# LOAD CALIBRATION
# ==========================================
with open(CALIB_FILE, "r") as f:
    data = json.load(f)

K = np.array(data["K"])
D = np.array(data["D"])

print("Loaded K:\n", K)
print("Loaded D:\n", D)

# ==========================================
# PARAMETERS
# ==========================================
BALANCE = 1.0
# 0.0 → no black borders (more crop)
# 1.0 → full FOV (more distortion at edges)

# ==========================================
# PROCESS IMAGES
# ==========================================
for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to load {filename}")
        continue

    h, w = img.shape[:2]

    # ==========================================
    # COMPUTE NEW CAMERA MATRIX
    # ==========================================
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=BALANCE
    )

    # ==========================================
    # UNDISTORT MAP
    # ==========================================
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # ==========================================
    # SAVE
    # ==========================================
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(out_path, undistorted)

    print(f"Saved: {out_path}")

print("Done.")