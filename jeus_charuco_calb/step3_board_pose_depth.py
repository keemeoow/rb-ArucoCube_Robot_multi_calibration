# step3_board_pose_depth.py

import cv2
import numpy as np
import os
from glob import glob

def rodrigues_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def pixel_to_3d(x, y, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth / 1000.0  # mm -> m
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

# Load calibration result
calib = np.load("/home/sstone/2025ZEUS/Calibration/charuco_calib_CG2_result.npz")
K = calib["camera_matrix"]
D = calib["dist_coeffs"]

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(
    squaresX=11,
    squaresY=7,
    squareLength=0.022,  # 22mm
    markerLength=0.016,  # 16mm
    dictionary=aruco_dict
)
# Image paths
image_dir = "/home/sstone/2025ZEUS/Calibration/Hand_in_Eye/board_image_CG2_1111/"
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

depth_paths = [p.replace(".png", "_depth.npy") for p in image_paths]

charuco_poses = []
reproj_errors = []
valid_indices = []

for idx, (rgb_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
    img = cv2.imread(rgb_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is None:
        print(f"[{idx}] No markers found.")
        continue

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if not ret or charuco_ids is None or len(charuco_ids) < 10:
        print(f"[{idx}] Not enough corners.")
        continue
    
    # subpixel 보정 추가
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if ret and charuco_ids is not None and len(charuco_ids) > 0:
        cv2.cornerSubPix(
            gray, 
            charuco_corners, 
            winSize=(5, 5), 
            zeroZone=(-1, -1), 
            criteria=criteria
        )
    # 보정 끝

    try:
        depth = np.load(depth_path)
    except:
        print(f"[{idx}] Depth file not found: {depth_path}")
        continue

    # 2D-3D correspondence
    img_points = []
    obj_points = []
    board_corners = board.chessboardCorners

    for i, cid in enumerate(charuco_ids.flatten()):
        x, y = charuco_corners[i][0]
        d = depth[int(round(y)), int(round(x))]
        if d == 0:
            continue

        xyz = pixel_to_3d(x, y, d, K)
        img_points.append([x, y])
        obj_points.append(board_corners[cid])

    if len(obj_points) < 6:
        print(f"[{idx}] Too few valid 3D points: {len(obj_points)}")
        continue

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        # Reprojection error
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        error = np.linalg.norm(projected.squeeze() - img_points, axis=1).mean()

        if error < 0.4:
            T = rodrigues_to_matrix(rvec, tvec)
            charuco_poses.append(T)
            reproj_errors.append(error)
            valid_indices.append(idx)
            print(f"[{idx}] OK - Reprojection error: {error:.4f} px")
        else:
            print(f"[{idx}] Reprojection error too high: {error:.4f} px → Skipped")
    else:
        print(f"[{idx}] solvePnP failed")

# 저장
np.save("charuco_poses_d.npy", np.array(charuco_poses))
np.save("charuco_pose_errors.npy", np.array(reproj_errors))
np.save("charuco_valid_indices.npy", np.array(valid_indices))

print(f"\nSuccessfully saved {len(charuco_poses)} poses.")
if reproj_errors:
    print(f"Mean reprojection error: {np.mean(reproj_errors):.4f} px")
