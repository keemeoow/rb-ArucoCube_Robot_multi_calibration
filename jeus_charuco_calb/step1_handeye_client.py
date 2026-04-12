"""
step1_handeye_client.py (zeus server랑 통신하면서 보드 촬영하여 저장하는 코드)
- 가상환경 : eyeinhand
- def capture_and_save() : 보드 촬영 및 저장
- def start_client() : 서버와 통신
"""

import socket
import json
import cv2
import re

import pyrealsense2 as rs
import numpy as np
import os
import time

# *** 서버 통신 정보 확인 (3번째 숫자의 0번, 1번 확인)
HOST = '192.168.0.23'
PORT = 12348

# **** 저장 폴더명 확인
save_dir = "/home/sprout/Desktop/**jiwoo/rb-multi_camera_calib/jeus/"
os.makedirs(save_dir, exist_ok=True)
frame_idx = 0

# **** ArUco/Charuco 보드 정의 확인
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(
    squaresX=11,
    squaresY=7,
    squareLength=0.022,  # 25mm
    markerLength=0.016,  # 18mm
    dictionary=aruco_dict
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)

device = profile.get_device()
color_sensor = device.query_sensors()[1]  # 0 = depth, 1 = color (대부분)
color_sensor.set_option(rs.option.enable_auto_exposure, 1)
color_sensor.set_option(rs.option.enable_auto_white_balance, 1)

spatial = rs.spatial_filter()       # 공간 필터 (엣지 보존)
temporal = rs.temporal_filter()     # 시간 필터 (노이즈 줄이기)
hole_filling = rs.hole_filling_filter()  # 홀 채우기

align_to = rs.stream.color
align = rs.align(align_to)

calib = np.load("/home/sprout/Desktop/**jiwoo/rb-multi_camera_calib/jeus/charuco_calib_result.npz")
camera_matrix = calib["camera_matrix"]
dist_coeffs = calib["dist_coeffs"]

params = cv2.aruco.DetectorParameters_create()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# joint_list = [
# [-18.90,  -9.09, -63.49,-179.97, 107.55, -16.85],
# [ -18.90, -16.08, -36.73,-179.96, 127.32, -16.84 ],
# [-18.90,  -8.39, -95.67,-179.97,  76.07, -16.87],
# [-12.48,  -8.12, -95.99,-179.95,  76.01, -10.45],
# [-6.48, -14.19, -88.34,-179.94,  77.60,  -4.45],
# [ -13.13, -26.00, -71.08,-179.96,  83.05, -11.10 ],
# [-8.75, -25.53, -87.91,-179.94,  66.68,  -6.73] ,
# [-8.74, -23.54, -82.84,-179.97,  73.76, -16.71] ,
# [-9.61, -16.31, -85.44,-180.02,  78.39, -37.56],
# [ -9.62, -16.30, -85.49,-179.89,  78.30,  22.40 ],
# [-9.80, -15.80, -91.52,-178.45,  62.90,  -8.30],
# [-9.44, -18.58, -86.87,-181.23,  84.60,  -7.11 ],
# [-8.05, -28.76, -63.61,-182.08, 107.64,  -6.22],
# [-11.68,  -4.90, -98.25,-176.06,  57.30, -11.15],
# [-2.78,  -8.39, -89.95,-190.04,  81.30,  -1.29] ,
# [ 4.04,  -7.15, -91.36,-192.72,  62.79,   9.77],
# [-25.11,  -8.85, -94.58,-170.78,  81.00, -26.91],
# [-24.86,  -9.31, -93.46,-172.85,  84.40, -46.01] ,
# [-25.18,  -7.87, -95.99,-169.87,  73.78,  12.19],
# [-15.19,  -4.86,-106.27,-179.93,  68.89, -15.22],
# [ -9.59, -31.57, -68.18,-179.93,  80.26,  -9.60 ],
# [-4.11, -31.87, -67.66,-179.93,  80.46,  -4.12 ],
# [-4.11, -36.26, -28.21,-179.93, 115.53,  -4.07],
# [-3.38, -19.76, -53.00,-190.34, 106.38,  -6.38],
# [9.44, -20.77, -49.36,-202.32,  97.78,   6.66 ],
# [ 8.88, -25.48, -42.11,-202.35, 128.09,  -5.25],
# [18.25, -31.88, -31.92,-200.40, 134.75,   3.52],
# [7.00, -26.05, -44.16,-189.84, 130.38,   0.22],
# [7.12, -22.56, -61.90,-189.00, 106.49,   4.52 ],
# [6.51, -29.34, -51.72,-188.81, 119.59,   1.80],
# [9.09, -11.59, -75.97,-194.99,  76.93,  12.40 ],
# [1.26, -12.82, -76.63,-184.10,  96.55, -19.11   ],
# [ 1.47, -16.21, -71.62,-185.91,  96.41, -39.09],
# [0,0,0,0,0,0]]


joint_list = [
[-18.90,  -9.09, -63.49,-179.97, 107.55, -16.85],
[ -18.90, -16.08, -36.73,-179.96, 127.32, -16.84 ],
[-5.90,  -8.52, -95.52,-179.94,  76.08,  -3.88],
[-13.22,  -8.10, -97.13,-175.90,  75.70, -12.22],
[-6.48, -14.19, -88.34,-179.94,  77.60,  -4.45],
[ -13.89, -27.42, -70.96,-172.99,  92.38, -11.50],
[ -8.75, -21.91, -61.86,-179.94,  96.34,  -6.70 ] ,
[-9.31, -21.88, -64.81,-175.18,  94.92, -16.92] ,
[-10.23, -15.89, -79.54,-175.31,  88.44, -38.47 ],
[ -9.62, -16.30, -85.49,-179.89,  78.30,  22.40 ],
[-10.02, -15.86, -83.76,-176.95,  80.93,  -8.42 ],
[-10.67, -18.51, -89.75,-171.49,  83.31,  -9.53],
[-8.17, -28.75, -63.90,-181.08, 107.47,  -6.02],
[ -11.89,  -4.69, -92.31,-174.91,  67.77, -11.37],
[ -4.58,  -8.15, -92.80,-179.93,  79.05,  -4.60 ] ,
[ 3.97,  -8.36, -60.60,-191.27,  91.71,   3.47],
[-25.11,  -8.85, -94.58,-170.78,  81.00, -26.91],
[-24.86,  -8.59, -86.52,-172.88,  91.99, -45.06] ,
[-25.18,  -7.87, -95.99,-169.87,  73.78,  12.19],
[-15.19,  -2.63, -77.24,-179.93, 100.15, -15.18 ],
[-9.59, -32.40, -43.37,-179.93, 104.23,  -9.57],
[-4.11, -31.42, -52.90,-179.93,  95.67,  -4.10],
[-4.11, -36.26, -28.21,-179.93, 115.53,  -4.07],
[-3.31, -25.95, -27.59,-192.17, 125.25, -10.46 ],
[ 8.51, -22.48, -42.70,-197.48, 105.06,   6.06],
[  8.20, -27.12, -35.14,-198.92, 133.22,  -5.07],
[18.25, -31.88, -31.92,-200.40, 134.75,   3.52],
[7.00, -26.05, -44.16,-189.84, 130.38,   0.22],
[7.12, -22.56, -61.90,-189.00, 106.49,   4.52 ],
[6.51, -29.34, -51.72,-188.81, 119.59,   1.80],
[8.98, -12.04, -70.82,-194.18,  84.33,  10.38],
[1.26, -12.82, -76.63,-184.10,  96.55, -19.11   ],
[ 1.47, -16.21, -71.62,-185.91,  96.41, -39.09],
[0,0,0,0,0,0]]


print("Waiting for server command...")

"""
capture_and_save(): 보드 촬영 및 저장
"""
def capture_and_save():
    global frame_idx

    # 안정화를 위한 dummy 프레임 drop
    for d in range(5):
        print(f"안정화를 위한 dummy_{d} 프레임 drop ")
        pipeline.wait_for_frames()

    depth_accum = []
    # ※ 이미 전역에 spatial/temporal/hole_filling, align 정의되어 있음

    frames = None
    for _ in range(5):
        frames = pipeline.wait_for_frames()
        # 컬러 기준으로 정렬
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        # 필터 적용 (정렬된 depth에)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        depth_accum.append(np.asanyarray(depth_frame.get_data()))

    # 평균 depth (float로 평균 후 uint16로)
    depth_avg = np.mean(depth_accum, axis=0).astype(np.uint16)

    # 마지막 aligned에서 color 꺼내기
    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Charuco 감지/포즈 추정 (동일)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is not None:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        if ret and charuco_corners is not None and len(charuco_ids) > 20:
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            valid = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec=rvec,
                tvec=tvec
            )
            if valid:
                rgb_path   = os.path.join(save_dir, f"{frame_idx:03d}.png")
                depth_path = os.path.join(save_dir, f"{frame_idx:03d}_depth.npy")
                cv2.imwrite(rgb_path, color_image)
                np.save(depth_path, depth_avg)
                print(f"[Saved] RGB: {rgb_path}, Depth(aligned): {depth_path}")
                frame_idx += 1

"""
start_client(): 서버와 통신
"""
def start_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")

        joint_idx = 0
        while joint_idx < len(joint_list):
            joint = joint_list[joint_idx]
            if all(x == 0 for x in joint):
                print("End of joint list.")
                break

            command = s.recv(1024).decode('utf-8')
            if command == 'capture':
                
                d1, d2, d3, d4, d5, d6 = joint
                result = {
                    'status': 'success',
                    'action': 'capture',
                    'd1': d1, 'd2': d2, 'd3': d3,
                    'd4': d4, 'd5': d5, 'd6': d6
                }
                s.sendall(json.dumps(result).encode('utf-8'))
                joint_idx += 1
                time.sleep(1.5)
                capture_and_save()
            elif command == 'quit':
                print("Received quit command.")
                break

    except socket.error as e:
        print("Socket error: {}".format(e))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        s.close()

if __name__ == '__main__':
    start_client()
