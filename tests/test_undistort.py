import cv2
import numpy as np
from pathlib import Path


K_OLD = np.array([
    [1062.5107262633112, 0.0, 959.2683916149307],
    [0.0, 1068.2607226108612, 766.5325292655051],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D_OLD = np.array([
    [-0.07711411357077246],
    [ 0.01781462873150522],
    [-0.025766365848438964],
    [ 0.006764936599292452],
], dtype=np.float64)


K_NEW = np.array([
    [998.4071233459787, 0.0, 965.5719167166222],
    [0.0, 1001.2218406185871, 774.6725758569834],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D_NEW = np.array([
    [-0.05107059136836853],
    [-0.019499344345137476],
    [ 0.017414830117291828],
    [-0.01020589295124726],
], dtype=np.float64)


def interpolate_calibration(alpha):

    K = (
        (1.0 - alpha) * K_OLD
        + alpha * K_NEW
    )

    D = (
        (1.0 - alpha) * D_OLD
        + alpha * D_NEW
    )

    return K, D


image = cv2.imread("/home/orin/Documents/PROJETO_ETICO/birds_eye_view/Fisheye_ChArUco_Calibration/data/teste/capture_0000_20260917_115102_forced_00corners.png")

h, w = image.shape[:2]

for alpha in np.arange(0.0, 1.01, 0.1):

    K, D = interpolate_calibration(alpha)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        (w, h),
        np.eye(3),
        balance=0.6,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        new_K,
        (w, h),
        cv2.CV_16SC2,
    )

    undistorted = cv2.remap(
        image,
        map1,
        map2,
        cv2.INTER_LINEAR,
    )

    cv2.imwrite(
        f"undistorted_alpha_{alpha:.1f}.png",
        undistorted,
    )