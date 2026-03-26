# Step2_to_capture_capture.py
"""
Step 2 (hand-to-eye): 그리퍼 카메라 + 고정 카메라를 이용한 Place-and-Capture 캘리브레이션.

파이프라인:
  1. 로봇이 ArUco 큐브를 작업 공간의 특정 위치에 놓음
  2. 로봇이 위로 이동하여 그리퍼 카메라가 큐브를 위에서 볼 수 있게 함
  3. 모든 카메라 (그리퍼 + 고정)가 동시에 촬영
  4. 마커별 및 큐브 전체 PnP를 계산하고 저장
  5. 로봇이 큐브를 집어 다음 위치로 이동
  6. 반복

정확한 변환 행렬 계산을 위해 다음을 활용:
  - 마커별 카메라-큐브 변환 (마커 1개만으로도 유효)
  - 로봇 기구학 (촬영 TCP 포즈 = 그리퍼 카메라 위치)
  - 그리퍼 + 고정 카메라 간 다시점 제약 조건

실행 명령어:
  python Step2_to_capture_capture.py \
    --root_folder ./data/session_v2 \
    --intrinsics_dir ./intrinsics \
    --use_robot \
    --robot_ip 192.168.0.23 \
    --robot_port 12348 \
    --waypoint_file new2_waypoints.json \
    --settle_time 1.5 \
    --save_depth \
    --show \
    --min_markers 1 \
    --min_cams_with_cube 1
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera import RealSenseCamera
from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt
from config import CubeConfig, CharucoBoardConfig
from robot_comm import PlaceCaptureClient, euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def annotate_image(bgr, cube, cam_idx, is_gripper, n_markers, ids, corners,
                    board_detector=None):
    """마커 오버레이 및 정보 텍스트를 이미지에 그림."""
    out = bgr.copy()

    # Gripper camera: draw board ArUco markers (DICT_4X4_250)
    n_board = 0
    if is_gripper and board_detector is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            bd_corners, bd_ids, _ = board_detector.detectMarkers(gray)
        except Exception:
            bd_corners, bd_ids = [], None
        if bd_ids is not None and len(bd_corners) > 0:
            n_board = len(bd_ids)
            try:
                cv2.aruco.drawDetectedMarkers(out, bd_corners, bd_ids)
            except Exception:
                pass

    # Draw cube markers
    if ids is not None and len(corners) > 0:
        try:
            draw_ids = ids.reshape(-1, 1) if getattr(ids, "ndim", 1) == 1 else ids
            cv2.aruco.drawDetectedMarkers(out, corners, draw_ids)
        except Exception:
            pass

    role = "GRIPPER" if is_gripper else "FIXED"
    ids_txt = ",".join(str(int(x)) for x in ids) if ids is not None and len(ids) > 0 else "-"
    board_txt = f" board={n_board}mkr" if is_gripper else ""
    lines = [
        f"cam{cam_idx} [{role}]",
        f"markers={n_markers} ids={ids_txt}{board_txt}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (4, y - 18), (10 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22
    return out


def filter_cube_markers(corners_list, ids, cube_ids_set):
    """Filter detection results to keep only cube marker IDs (0~4).
    Board markers (DICT_4X4_250, ID 5+) share patterns with DICT_4X4_50."""
    if ids is None or len(ids) == 0:
        return [], None
    filt_c, filt_i = [], []
    for c, mid in zip(corners_list, ids):
        if int(mid) in cube_ids_set:
            filt_c.append(c)
            filt_i.append(int(mid))
    if not filt_i:
        return [], None
    return filt_c, np.array(filt_i)


def make_quad_image(frames_dict, cam_order, cube, gripper_cam_idx, board_detector=None):
    """4개 카메라로부터 마커 오버레이가 포함된 2x2 분할 이미지를 생성."""
    tiles = []
    tile_h, tile_w = None, None

    for ci in cam_order:
        fr = frames_dict.get(ci)
        if fr is not None and fr.get("color") is not None:
            img = fr["color"]
            if tile_h is None:
                tile_h, tile_w = img.shape[:2]
            annotated = annotate_image(
                img, cube, ci,
                is_gripper=(ci == gripper_cam_idx),
                n_markers=fr.get("n_markers", 0),
                ids=fr.get("ids_np"),
                corners=fr.get("corners", []),
                board_detector=board_detector,
            )
            tiles.append(annotated)
        else:
            if tile_h is None:
                tile_h, tile_w = 480, 640
            blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.putText(blank, f"cam{ci} N/A", (20, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            tiles.append(blank)

    while len(tiles) < 4:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    tiles = tiles[:4]

    top = cv2.hconcat([tiles[0], tiles[1]])
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    return cv2.vconcat([top, bottom])


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    gripper_cam_idx = m.get("gripper_cam_idx", None)
    return serial_to_idx, gripper_cam_idx, map_path


def load_intrinsics(intr_dir: str, cam_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """카메라 내부 파라미터 행렬 K와 왜곡 계수 D를 로드."""
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Intrinsics not found: {p}")
    d = np.load(p, allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64)
    return K, D


def estimate_per_marker_poses(
    cube: ArucoCubeTarget,
    corners_list: list,
    ids: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> List[dict]:
    """
    알려진 큐브 형상을 이용하여 개별 마커의 포즈를 추정.
    마커 1개만으로도 카메라-큐브 변환을 추정할 수 있음.

    마커별 결과 리스트 (rvec, tvec, 재투영 오차 포함)를 반환.
    """
    results = []
    if ids is None or len(ids) == 0:
        return results

    for c, mid in zip(corners_list, ids):
        mid = int(mid)
        if mid not in cube.cfg.id_to_face:
            continue

        obj_pts = cube.model.marker_corners_in_rig(mid)  # (4, 3) 큐브 좌표계 기준
        img_pts = c.reshape(4, 2).astype(np.float64)

        # 마커 3번의 코너 순서 재정렬 (aruco_cube.py 규칙에 맞춤)
        if mid == 3:
            img_pts = img_pts[[1, 2, 3, 0]]

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts.reshape(-1, 1, 3).astype(np.float64),
            img_pts.reshape(-1, 1, 2).astype(np.float64),
            K, D,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not ok:
            continue

        # 재투영 오차 계산
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)

        # 단일 마커로부터 카메라-큐브 변환 행렬 생성
        T_cam_cube = rodrigues_to_Rt(rvec, tvec)

        results.append({
            "marker_id": mid,
            "face": cube.cfg.id_to_face[mid],
            "corners_2d": img_pts.tolist(),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "reproj_error_mean_px": float(np.mean(err)),
            "reproj_error_max_px": float(np.max(err)),
            "T_cam_cube_4x4": T_cam_cube.tolist(),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Place-and-Capture calibration: gripper camera + fixed cameras"
    )
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)

    # 스트림 설정
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    # 검출 설정
    parser.add_argument("--min_markers", type=int, default=1,
                        help="Min markers per camera to count as 'cube visible'")
    parser.add_argument("--min_cams_with_cube", type=int, default=1,
                        help="Min cameras that must see cube to accept capture")

    # 뎁스 저장
    parser.add_argument("--save_depth", action="store_true")

    # 화면 표시
    parser.add_argument("--show", action="store_true")

    # 로봇 모드
    parser.add_argument("--use_robot", action="store_true")
    parser.add_argument("--robot_ip", type=str, default="192.168.0.23")
    parser.add_argument("--robot_port", type=int, default=12348)
    parser.add_argument("--waypoint_file", type=str, default=None,
                        help="JSON file with list of {place, capture} waypoint pairs")
    parser.add_argument("--manual_robot", action="store_true",
                        help="Manual robot mode: server sends capture commands interactively (use with robot_calb.py)")
    parser.add_argument("--settle_time", type=float, default=1.5,
                        help="Wait time (s) after robot signals capture before taking images")

    args = parser.parse_args()

    root = ensure_dir(args.root_folder)
    intr_dir = args.intrinsics_dir

    # ─── 디바이스 맵 로드 ───
    serial_to_idx, gripper_cam_idx, _ = load_device_map(intr_dir)
    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found.")

    if serial_to_idx is None:
        print("[WARN] No device_map.json. Run Step1 first.")
        serials = sorted(devs.keys())
        idx_serial_pairs = [(i, s) for i, s in enumerate(serials)]
        gripper_cam_idx = None
    else:
        idx_serial_pairs = []
        for serial in devs.keys():
            if serial in serial_to_idx:
                idx_serial_pairs.append((int(serial_to_idx[serial]), serial))
        idx_serial_pairs.sort(key=lambda x: x[0])

    if len(idx_serial_pairs) == 0:
        raise RuntimeError("No usable cameras found.")

    n_fixed = 0
    n_gripper = 0
    print("[INFO] Cameras:")
    for idx, s in idx_serial_pairs:
        if idx == gripper_cam_idx:
            tag = "GRIPPER"
            n_gripper += 1
        else:
            tag = "FIXED"
            n_fixed += 1
        print(f"  cam{idx}: {s} ({tag})")

    if gripper_cam_idx is None:
        print("[WARN] No gripper camera configured in device_map.json.")
        print("[WARN] Gripper camera views will not be available.")
    else:
        print(f"[INFO] Gripper camera: cam{gripper_cam_idx}")

    print(f"[INFO] Fixed cameras: {n_fixed}, Gripper cameras: {n_gripper}")

    # ─── PnP용 내부 파라미터 로드 ───
    cam_intrinsics: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for ci, _ in idx_serial_pairs:
        try:
            K, D = load_intrinsics(intr_dir, ci)
            cam_intrinsics[ci] = (K, D)
            print(f"[INFO] Loaded intrinsics for cam{ci}")
        except FileNotFoundError:
            print(f"[WARN] No intrinsics for cam{ci}. Per-marker PnP will be skipped.")

    # ─── 카메라 시작 ───
    cams: Dict[int, RealSenseCamera] = {}
    for ci, serial in idx_serial_pairs:
        cam = RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_color=True,
            use_depth=args.save_depth,
            align_depth_to_color=True,
            warmup_frames=10,
        )
        cam.start()
        cams[ci] = cam
        ensure_dir(os.path.join(root, f"cam{ci}"))

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)
    _cube_ids = set(cfg.marker_ids)  # {0,1,2,3,4} — filter out board markers

    # Board marker detector (DICT_4X4_250) — gripper camera visualization only
    _board_cfg = CharucoBoardConfig()
    _board_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, _board_cfg.dictionary_name))
    try:
        _board_det = cv2.aruco.ArucoDetector(
            _board_dict, cv2.aruco.DetectorParameters())
    except AttributeError:
        _board_det = None

    # ─── 웨이포인트 로드 ───
    waypoint_list: List[dict] = []
    if args.waypoint_file:
        with open(args.waypoint_file, "r") as f:
            waypoint_list = json.load(f)
        print(f"[INFO] Loaded {len(waypoint_list)} waypoints from {args.waypoint_file}")

    # ─── 로봇 클라이언트 ───
    robot_client: Optional[PlaceCaptureClient] = None
    if args.use_robot and not args.manual_robot:
        robot_client = PlaceCaptureClient(args.robot_ip, args.robot_port)
        robot_client.connect()

    # ─── 메타 데이터 ───
    meta = {
        "root_folder": os.path.abspath(root),
        "gripper_cam_idx": gripper_cam_idx,
        "n_fixed_cams": n_fixed,
        "n_gripper_cams": n_gripper,
        "cam_indices": [ci for ci, _ in idx_serial_pairs],
        "captures": [],
    }
    meta_path = os.path.join(root, "meta.json")
    quad_dir = ensure_dir(os.path.join(root, "marker_quads"))
    cam_order = sorted(ci for ci, _ in idx_serial_pairs)
    event_id = 0

    print("\nControls:")
    print("  SPACE : manual capture (if in manual mode)")
    print("  ESC/q : quit\n")

    def do_capture(
        capture_pose_6dof: Optional[List[float]] = None,
        place_pose_6dof: Optional[List[float]] = None,
        pose_index: Optional[int] = None,
        grip_target_tvec: Optional[List[float]] = None,
    ) -> bool:
        """모든 카메라에서 마커별 포즈 추정과 함께 촬영."""
        nonlocal event_id

        # 안정화 대기
        if args.settle_time > 0 and args.use_robot:
            time.sleep(args.settle_time)

        frames: Dict[int, dict] = {}
        cams_with_cube = 0

        for ci, cam in cams.items():
            color, depth, ts_ms = cam.get_latest()
            if color is None:
                continue

            corners, ids = cube.detect(color)
            # Fixed cameras: cube markers only (exclude board markers)
            if ci != gripper_cam_idx:
                corners, ids = filter_cube_markers(corners, ids, _cube_ids)
            n_markers = 0 if ids is None else len(ids)
            ok = n_markers >= args.min_markers
            if ok:
                cams_with_cube += 1

            # 마커별 PnP
            marker_poses = []
            cube_pnp = None
            if ci in cam_intrinsics and ids is not None and len(ids) > 0:
                K, D = cam_intrinsics[ci]

                # 개별 마커 포즈 추정
                marker_poses = estimate_per_marker_poses(cube, corners, ids, K, D)

                # 큐브 전체 PnP (보이는 모든 마커 사용)
                pnp_ok, rvec, tvec, used_ids, reproj = cube.solve_pnp_cube(
                    color, K, D,
                    use_ransac=True,
                    min_markers=1,
                    return_reproj=True,
                )
                if pnp_ok and rvec is not None:
                    T_cam_cube = rodrigues_to_Rt(rvec, tvec)
                    cube_pnp = {
                        "ok": True,
                        "rvec": rvec.flatten().tolist(),
                        "tvec": tvec.flatten().tolist(),
                        "used_ids": [int(x) for x in used_ids],
                        "reproj_mean_px": reproj["err_mean"] if reproj else None,
                        "T_cam_cube_4x4": T_cam_cube.tolist(),
                    }

            frames[ci] = {
                "color": color,
                "depth": depth,
                "ts_ms": ts_ms,
                "ok": ok,
                "n_markers": n_markers,
                "ids": ([] if ids is None else [int(x) for x in ids]),
                "corners": corners,
                "ids_np": ids,
                "marker_poses": marker_poses,
                "cube_pnp": cube_pnp,
            }

        if cams_with_cube < args.min_cams_with_cube:
            print(f"[SKIP] Only {cams_with_cube}/{args.min_cams_with_cube} cams see cube.")
            return False

        # ─── 저장 ───
        fid = int(event_id)
        cap_rec: dict = {
            "event_id": fid,
            "pose_index": pose_index,
            "cams": {},
        }

        # 로봇 포즈 데이터
        # capture_pose = 이미지 촬영 시 현재 로봇 TCP
        # Step3에서 참조: robot_pose_6dof / robot_pose_matrix_4x4
        robot_tcp = capture_pose_6dof or place_pose_6dof
        if robot_tcp is not None:
            tcp_f = [float(x) for x in robot_tcp]
            cap_rec["robot_pose_6dof"] = tcp_f        # Step3 compatible
            cap_rec["capture_pose_6dof"] = tcp_f      # new2 format
            try:
                T44 = euler_deg_to_matrix(*tcp_f).tolist()
                cap_rec["robot_pose_matrix_4x4"] = T44  # Step3 compatible
                cap_rec["capture_pose_matrix_4x4"] = T44
            except Exception:
                pass

        if grip_target_tvec is not None:
            cap_rec["grip_target_tvec"] = [float(x) for x in grip_target_tvec]

        if place_pose_6dof is not None and place_pose_6dof != robot_tcp:
            cap_rec["place_pose_6dof"] = [float(x) for x in place_pose_6dof]
            try:
                cap_rec["place_pose_matrix_4x4"] = euler_deg_to_matrix(
                    *place_pose_6dof
                ).tolist()
            except Exception:
                pass

        for ci in sorted(frames.keys()):
            fr = frames[ci]

            rgb_rel = f"cam{ci}/rgb_{fid:05d}.jpg"
            cv2.imwrite(os.path.join(root, rgb_rel), fr["color"])

            depth_rel = None
            if args.save_depth and fr["depth"] is not None:
                depth_rel = f"cam{ci}/depth_{fid:05d}.png"
                cv2.imwrite(os.path.join(root, depth_rel), fr["depth"])

            cam_rec = {
                "saved": True,
                "is_gripper": (ci == gripper_cam_idx),
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "ts_ms": fr["ts_ms"],
                "n_markers_detected": fr["n_markers"],
                "marker_ids": fr["ids"],
                "cube_visible": fr["ok"],
                "markers": fr["marker_poses"],  # per-marker PnP results
            }

            if fr["cube_pnp"] is not None:
                cam_rec["cube_pnp"] = fr["cube_pnp"]

            cap_rec["cams"][str(ci)] = cam_rec

        meta["captures"].append(cap_rec)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 마커 오버레이가 포함된 2x2 분할 이미지 저장
        quad = make_quad_image(frames, cam_order, cube, gripper_cam_idx, _board_det)
        quad_path = os.path.join(quad_dir, f"frame_{fid:05d}.jpg")
        cv2.imwrite(quad_path, quad)

        # 분할 이미지 표시
        if args.show:
            cv2.imshow("Capture Quad", quad)
            cv2.waitKey(500)

        # 요약 출력
        cam_summary = []
        for ci in sorted(frames.keys()):
            fr = frames[ci]
            tag = "G" if ci == gripper_cam_idx else "F"
            n = fr["n_markers"]
            cam_summary.append(f"cam{ci}({tag}):{n}mkr")
        print(f"[SAVE] event={fid} | {' '.join(cam_summary)} | quad={quad_path}")
        event_id += 1
        return True

    try:
        if args.use_robot and args.manual_robot:
            # ─── 수동 로봇 모드 (robot_calb.py 서버 사용) ───
            print("[MODE] Manual Robot - waiting for server capture commands")
            print("[INFO] Move robot on server side, press 'c' to capture\n")

            import socket as _sock
            import threading

            manual_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
            manual_sock.settimeout(None)
            manual_sock.connect((args.robot_ip, args.robot_port))
            print(f"[ManualRobot] Connected to {args.robot_ip}:{args.robot_port}")

            # ─── Live preview thread ───
            preview_running = True

            def preview_loop():
                while preview_running:
                    live_frames = {}
                    for ci, cam in cams.items():
                        color, _, _ = cam.get_latest()
                        if color is None:
                            continue
                        corners, ids = cube.detect(color)
                        # Fixed cameras: cube markers only (exclude board markers)
                        if ci != gripper_cam_idx:
                            corners, ids = filter_cube_markers(corners, ids, _cube_ids)
                        n = 0 if ids is None else len(ids)
                        live_frames[ci] = {
                            "color": color,
                            "n_markers": n,
                            "ids_np": ids,
                            "corners": corners,
                        }

                    if live_frames:
                        quad = make_quad_image(live_frames, cam_order, cube, gripper_cam_idx, _board_det)
                        # Resize for preview
                        ph = int(quad.shape[0] * 0.6)
                        pw = int(quad.shape[1] * 0.6)
                        preview = cv2.resize(quad, (pw, ph))
                        cv2.imshow("Live Preview (4 cameras)", preview)

                    key = cv2.waitKey(100) & 0xFF
                    if key == 27 or key == ord('q'):
                        break

            if args.show:
                preview_thread = threading.Thread(target=preview_loop, daemon=True)
                preview_thread.start()
                print("[INFO] Live preview started (4-camera quad view)")

            try:
                while True:
                    data = manual_sock.recv(8192)
                    if not data:
                        print("[ManualRobot] Server disconnected.")
                        break

                    msg = json.loads(data.decode("utf-8").strip())
                    cmd = msg.get("command", "")

                    if cmd == "quit":
                        print("[ManualRobot] Server sent quit.")
                        break

                    if cmd == "capture":
                        capture_tcp = msg.get("capture_pose_6dof")
                        pose_idx = msg.get("pose_index", event_id)
                        g_tvec = msg.get("grip_target_tvec")

                        print(f"\n[ManualRobot] Capture signal received (pose_index={pose_idx})")
                        if capture_tcp:
                            print(f"  TCP: {capture_tcp}")
                        if g_tvec:
                            print(f"  grip_tvec: [{g_tvec[0]:.4f}, {g_tvec[1]:.4f}, {g_tvec[2]:.4f}]")

                        saved = do_capture(
                            capture_pose_6dof=capture_tcp,
                            pose_index=pose_idx,
                            grip_target_tvec=g_tvec,
                        )

                        status = "success" if saved else "skipped"
                        resp = json.dumps({"action": "captured", "status": status})
                        manual_sock.sendall(resp.encode("utf-8"))

                        if saved:
                            print(f"[OK] Capture {pose_idx} saved")
                        else:
                            print(f"[SKIP] Capture {pose_idx} skipped")
                    elif cmd == "detect":
                        # Visual servoing: detect cube from gripper camera
                        if gripper_cam_idx is None or gripper_cam_idx not in cams:
                            resp = json.dumps({"ok": False, "reason": "no_gripper_cam"})
                            manual_sock.sendall(resp.encode("utf-8"))
                            continue
                        if gripper_cam_idx not in cam_intrinsics:
                            resp = json.dumps({"ok": False, "reason": "no_intrinsics"})
                            manual_sock.sendall(resp.encode("utf-8"))
                            continue

                        g_color, _, _ = cams[gripper_cam_idx].get_latest()
                        if g_color is None:
                            resp = json.dumps({"ok": False, "reason": "no_image"})
                            manual_sock.sendall(resp.encode("utf-8"))
                            continue

                        g_K, g_D = cam_intrinsics[gripper_cam_idx]
                        det_ok, det_rv, det_tv, det_used = cube.solve_pnp_cube(
                            g_color, g_K, g_D, use_ransac=False, min_markers=1,
                            reproj_thr_mean_px=10.0)

                        if det_ok:
                            resp = json.dumps({
                                "ok": True,
                                "tvec": det_tv.flatten().tolist(),
                                "rvec": det_rv.flatten().tolist(),
                                "used_ids": [int(x) for x in det_used],
                            })
                            print(f"[Detect] tvec=[{det_tv[0][0]:.4f}, {det_tv[1][0]:.4f}, {det_tv[2][0]:.4f}] ids={det_used}")
                        else:
                            resp = json.dumps({"ok": False, "reason": "detection_failed",
                                               "n_markers": len(det_used) if det_used else 0})
                            print(f"[Detect] Failed (markers={det_used})")
                        manual_sock.sendall(resp.encode("utf-8"))
                    else:
                        print(f"[ManualRobot] Unknown command: {cmd}")

            finally:
                preview_running = False
                manual_sock.close()

            print(f"\n[DONE] Manual robot capture complete. {event_id} captures saved.")

        elif args.use_robot and waypoint_list:
            # ─── Robot Place-and-Capture mode ───
            print("[MODE] Robot Place-and-Capture")
            print(f"[INFO] {len(waypoint_list)} waypoints to process\n")

            for wi, wp in enumerate(waypoint_list):
                place_pose = wp["place"]
                capture_pose = wp["capture"]

                print(f"\n[Robot] Waypoint {wi+1}/{len(waypoint_list)}")
                print(f"  Place:   {place_pose}")
                print(f"  Capture: {capture_pose}")

                # Run the full protocol cycle
                try:
                    ok, actual_capture_tcp, actual_place_tcp = \
                        robot_client.run_single_waypoint(place_pose, capture_pose)
                except Exception as e:
                    print(f"[ERROR] Robot communication error: {e}")
                    break

                if not ok:
                    print("[INFO] Robot quit or error.")
                    break

                # Capture from all cameras
                saved = do_capture(
                    capture_pose_6dof=actual_capture_tcp,
                    place_pose_6dof=actual_place_tcp,
                    pose_index=wi,
                )

                # Acknowledge to server (so it can pick up cube)
                try:
                    robot_client.acknowledge_capture()
                except Exception as e:
                    print(f"[ERROR] Failed to acknowledge: {e}")
                    break

                if saved:
                    print(f"[OK] Waypoint {wi+1} captured successfully")
                else:
                    print(f"[WARN] Waypoint {wi+1} skipped (not enough markers visible)")

            # Send quit to server
            try:
                robot_client.wait_for_ready()
                robot_client.send_quit()
            except Exception:
                pass

            print(f"\n[DONE] Robot capture complete. {event_id} captures saved.")

        else:
            # ─── Manual mode ───
            print("[MODE] Manual capture (press SPACE)")
            while True:
                frames_view: Dict[int, dict] = {}
                for ci, cam in cams.items():
                    color, _, _ = cam.get_latest()
                    if color is None:
                        continue

                    corners, ids = cube.detect(color)
                    ok = (ids is not None) and (len(ids) >= args.min_markers)
                    frames_view[ci] = {
                        "color": color, "ok": ok,
                        "corners": corners, "ids_np": ids,
                    }

                if args.show:
                    for ci in sorted(frames_view.keys()):
                        img = frames_view[ci]["color"].copy()
                        ids_np = frames_view[ci]["ids_np"]
                        corners = frames_view[ci]["corners"]
                        if ids_np is not None:
                            try:
                                draw_ids = ids_np.reshape(-1, 1) if getattr(ids_np, "ndim", 1) == 1 else ids_np
                                cv2.aruco.drawDetectedMarkers(img, corners, draw_ids)
                            except Exception:
                                pass
                        tag = "GRIP" if ci == gripper_cam_idx else "FIX"
                        n = 0 if ids_np is None else len(ids_np)
                        txt = f"cam{ci}({tag}) markers={n} ok={frames_view[ci]['ok']}"
                        cv2.putText(img, txt, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow(f"cam{ci}", img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                if key == 32:  # SPACE
                    do_capture()

    finally:
        for cam in cams.values():
            cam.stop()
        if robot_client:
            robot_client.close()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Total captures: {event_id}")
    print(f"  Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
