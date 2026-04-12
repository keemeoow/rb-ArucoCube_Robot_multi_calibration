# step2_charuo_intricsic_depth.py

import cv2
import numpy as np
import os
from glob import glob

# 보드 설정
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(
    squaresX=11,
    squaresY=7,
    squareLength=0.022,  # 25mm
    markerLength=0.016,  # 18mm
    dictionary=aruco_dict
)

# 이미지 경로
image_dir = "/home/sstone/2025ZEUS/Calibration/Hand_in_Eye/board_image_CG2_1111/"
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

# 코너 저장용 리스트
all_corners = []
all_ids = []
image_size = None

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None and len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ret and charuco_ids is not None and charuco_corners is not None:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            if image_size is None:
                image_size = gray.shape[::-1]

# === 1차 캘리브레이션 ===
flags = cv2.CALIB_RATIONAL_MODEL
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=flags
)

print("\n=== 1st Calibration ===")
print("Initial Reprojection Error:", ret)

# === Reprojection Error 계산 및 이상치 제거 ===
errors = []
for i in range(len(all_corners)):
    imgpoints2, _ = cv2.projectPoints(
        board.chessboardCorners[all_ids[i].flatten()],
        rvecs[i], tvecs[i],
        camera_matrix, dist_coeffs
    )
    error = np.linalg.norm(imgpoints2.squeeze() - all_corners[i].squeeze(), axis=1).mean()
    errors.append(error)

threshold = np.mean(errors) + np.std(errors)
filtered_corners = []
filtered_ids = []
for i, err in enumerate(errors):
    if err < threshold:
        filtered_corners.append(all_corners[i])
        filtered_ids.append(all_ids[i])

# === 2차 캘리브레이션 (이상치 제거 후) ===
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=filtered_corners,
    charucoIds=filtered_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=flags
)

# 결과 출력
print("\n=== Final Calibration Result ===")
print("Reprojection Error:", ret)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs.flatten())

# 저장
np.savez("/home/sstone/2025ZEUS/Calibration/charuco_calib_CG2_result.npz", 
         camera_matrix=camera_matrix, 
         dist_coeffs=dist_coeffs, 
         reproj_error=ret)
print("\nCalibration saved to charuco_calib_result.npz")
