# Step2_in_capture_capture.py
"""
Step 2: hand-in-eye (그리퍼 카메라) 캘리브레이션을 위한 ChArUco 보드 촬영.
  --also_detect_cube 옵션 시, 고정 카메라에서 ArUco 큐브도 동시 검출.

파이프라인:
  1. ChArUco 보드를 테이블에 고정 배치 (+ ArUco 큐브도 옆에 배치)
  2. 로봇이 그리퍼 카메라를 보드 위 다양한 자세로 이동
  3. 각 자세에서: 이미지 촬영 + 로봇 TCP 기록
  4. 정확한 Hand-eye 캘리브레이션을 위해 회전 다양성 필수

실행 명령어:
  서버 (로봇 컨트롤러):
    python robot_calb.py

  클라이언트 (컴퓨터):
    python Step2_in_capture_capture.py \
      --root_folder ./data/charuco_session \
      --intrinsics_dir ./intrinsics \
      --gripper_cam_idx 2 \
      --robot_ip 192.168.0.23 --robot_port 12348 \
      --show \
      --also_detect_cube
"""

import os
import json
import time
import argparse
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from camera import RealSenseCamera
from charuco_utils import CharucoTarget
from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt
from config import CharucoBoardConfig, CubeConfig
from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    return m.get("serial_to_idx", {}), m.get("gripper_cam_idx")


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def annotate_gripper(bgr, K, D, cam_idx, charuco, cube_det,
                     charuco_result=None, cube_result=None):
    """Draw ChArUco + ArUco cube overlay on gripper camera image.

    charuco_result: (ok, rvec, tvec, n_corners, reproj, charuco_corners, charuco_ids)
    cube_result:    (pnp_ok, rvec, tvec, cube_corners_list, cube_ids)
    If None, detect from bgr on the fly (for live preview).
    """
    out = bgr.copy()

    # --- ChArUco detection ---
    if charuco_result is not None:
        ch_ok, ch_rvec, ch_tvec, n_corners, reproj, ch_corners, ch_ids = charuco_result
    else:
        ch_corners, ch_ids, n_corners, _, _ = charuco.detect(bgr)
        ch_ok, ch_rvec, ch_tvec = False, None, None
        reproj = None
        if ch_corners is not None and n_corners >= 4:
            ch_ok, ch_rvec, ch_tvec, n_corners, reproj = charuco.estimate_pose(bgr, K, D)

    if ch_corners is not None and ch_ids is not None:
        try:
            cv2.aruco.drawDetectedCornersCharuco(out, ch_corners, ch_ids)
        except Exception:
            pass
    if ch_ok and ch_rvec is not None:
        try:
            cv2.drawFrameAxes(out, K, D, ch_rvec, ch_tvec, 0.05)
        except Exception:
            pass

    # --- ArUco cube detection ---
    if cube_result is not None:
        cu_ok, cu_rvec, cu_tvec, cu_corners, cu_ids = cube_result
    else:
        cu_corners, cu_ids = cube_det.detect(bgr)
        cu_ok, cu_rvec, cu_tvec = False, None, None
        if cu_ids is not None and len(cu_ids) >= 1:
            pnp_ok, rv, tv, _, _ = cube_det.solve_pnp_cube(
                bgr, K, D, use_ransac=False, min_markers=1,
                reproj_thr_mean_px=6.0, return_reproj=True,
            )
            if pnp_ok:
                cu_ok, cu_rvec, cu_tvec = True, rv, tv

    if cu_ids is not None and len(cu_corners) > 0:
        try:
            draw_ids = cu_ids.reshape(-1, 1) if getattr(cu_ids, "ndim", 1) == 1 else cu_ids
            cv2.aruco.drawDetectedMarkers(out, cu_corners, draw_ids)
        except Exception:
            pass
    if cu_ok and cu_rvec is not None:
        try:
            cv2.drawFrameAxes(out, K, D, cu_rvec, cu_tvec, 0.03)
        except Exception:
            pass

    # --- Text overlay ---
    reproj_txt = "reproj=N/A"
    if reproj is not None:
        try:
            reproj_txt = f"reproj={float(reproj):.2f}px" if float(reproj) < 1000 else "reproj=N/A"
        except Exception:
            pass

    n_cube = 0 if cu_ids is None else len(cu_ids)
    lines = [
        f"cam{cam_idx} [GRIPPER]",
        f"charuco={n_corners} {reproj_txt}",
        f"cube={n_cube}mkr",
    ]

    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (4, y - 18), (10 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22

    return out

def safe_estimate_charuco(charuco, color, K, D):
    try:
        return charuco.estimate_pose(color, K, D)
    except Exception as e:
        print(f"[WARN] ChArUco estimate failed: {e}")
        return False, None, None, 0, None

def main():
    parser = argparse.ArgumentParser(
        description="ChArUco board capture for eye-in-hand calibration"
    )
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--gripper_cam_idx", type=int, default=2)

    parser.add_argument("--robot_ip", type=str, default="192.168.0.23")
    parser.add_argument("--robot_port", type=int, default=12348)
    parser.add_argument("--settle_time", type=float, default=1.0)

    parser.add_argument("--min_corners", type=int, default=6,
                        help="Minimum ChArUco corners to accept capture")

    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save_depth", action="store_true")

    # Also detect ArUco cube from fixed cameras
    parser.add_argument("--also_detect_cube", action="store_true",
                        help="Also capture from fixed cameras and detect ArUco cube")
    parser.add_argument("--min_markers", type=int, default=1,
                        help="Min markers for cube detection on fixed cameras")

    # Stream config
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    args = parser.parse_args()

    root = ensure_dir(args.root_folder)
    gi = args.gripper_cam_idx

    # Load device map
    serial_to_idx, _ = load_device_map(args.intrinsics_dir)
    devs = RealSenseCamera.list_devices()

    # Build camera list
    idx_serial_pairs = []
    if serial_to_idx:
        for serial, idx_str in serial_to_idx.items():
            if serial in devs:
                idx_serial_pairs.append((int(idx_str), serial))
        idx_serial_pairs.sort(key=lambda x: x[0])

    gripper_serial = None
    fixed_cam_ids = []
    for idx, serial in idx_serial_pairs:
        if idx == gi:
            gripper_serial = serial
        else:
            fixed_cam_ids.append(idx)

    if gripper_serial is None:
        raise RuntimeError(f"Gripper camera (cam{gi}) not found in connected devices.")

    print(f"[INFO] Gripper camera: cam{gi} ({gripper_serial})")

    # Load intrinsics for all cameras
    K_map: Dict[int, np.ndarray] = {}
    D_map: Dict[int, np.ndarray] = {}
    K_g, D_g = load_intrinsics(args.intrinsics_dir, gi)
    K_map[gi] = K_g
    D_map[gi] = D_g
    print(f"[INFO] Intrinsics loaded for cam{gi} (gripper)")

    if args.also_detect_cube:
        for ci in fixed_cam_ids:
            try:
                K_map[ci], D_map[ci] = load_intrinsics(args.intrinsics_dir, ci)
                print(f"[INFO] Intrinsics loaded for cam{ci} (fixed)")
            except FileNotFoundError:
                print(f"[WARN] No intrinsics for cam{ci}, skipping")

    # Start cameras
    cams: Dict[int, RealSenseCamera] = {}
    cams_to_start = [(gi, gripper_serial)]
    if args.also_detect_cube:
        for idx, serial in idx_serial_pairs:
            if idx != gi:
                cams_to_start.append((idx, serial))

    for ci, serial in cams_to_start:
        c = RealSenseCamera(
            serial=serial,
            width=args.width, height=args.height, fps=args.fps,
            use_color=True, use_depth=args.save_depth,
            align_depth_to_color=True, warmup_frames=10,
        )
        c.start()
        cams[ci] = c
        ensure_dir(os.path.join(root, f"cam{ci}"))
        tag = "GRIPPER" if ci == gi else "FIXED"
        print(f"[INFO] cam{ci} started ({tag})")

    # Shortcut for gripper camera
    cam = cams[gi]
    K, D = K_g, D_g

    # ChArUco target
    charuco_cfg = CharucoBoardConfig()
    charuco = CharucoTarget(charuco_cfg)
    print(f"[INFO] ChArUco board: {charuco_cfg.squares_x}x{charuco_cfg.squares_y}, "
          f"square={charuco_cfg.square_length_m*1000:.0f}mm, "
          f"marker={charuco_cfg.marker_length_m*1000:.0f}mm, "
          f"marker_id_start={charuco_cfg.marker_id_start}")
    
    def safe_estimate_charuco(charuco, color, K, D):
        try:
            return charuco.estimate_pose(color, K, D)
        except Exception as e:
            print(f"[WARN] ChArUco estimate failed: {e}")
            return False, None, None, 0, None

    # ArUco cube target (gripper always detects cube + fixed cameras when enabled)
    cube_cfg = CubeConfig()
    cube = ArucoCubeTarget(cube_cfg)
    print(f"[INFO] ArUco cube: side={cube_cfg.cube_side_m*1000:.0f}mm, "
          f"marker={cube_cfg.marker_size_m*1000:.0f}mm")

    # Meta
    meta = {
        "root_folder": os.path.abspath(root),
        "calibration_type": "charuco_eye_in_hand",
        "gripper_cam_idx": gi,
        "charuco_config": {
            "squares_x": charuco_cfg.squares_x,
            "squares_y": charuco_cfg.squares_y,
            "square_length_m": charuco_cfg.square_length_m,
            "marker_length_m": charuco_cfg.marker_length_m,
            "dictionary": charuco_cfg.dictionary_name,
            "marker_id_start": charuco_cfg.marker_id_start,
        },
        "captures": [],
    }
    meta_path = os.path.join(root, "meta_charuco.json")
    event_id = 0

    # Connect to robot server
    import socket as _sock
    sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    sock.settimeout(3.0)
    try:
        sock.connect((args.robot_ip, args.robot_port))
    except ConnectionRefusedError:
        raise RuntimeError(
            f"Robot server refused connection: {args.robot_ip}:{args.robot_port}. "
            f"서버 미실행 / 포트 불일치 / 로봇 에러상태 가능성"
        )
    except TimeoutError:
        raise RuntimeError(
            f"Robot server timeout: {args.robot_ip}:{args.robot_port}. "
            f"IP 오타 / 네트워크 문제 가능성"
        )
    finally:
        sock.settimeout(None)
    print(f"\n[MODE] ChArUco Eye-in-Hand Capture")
    print(f"[INFO] Move gripper camera over the ChArUco board")
    print(f"[INFO] Press 'c' on server to capture, vary ROTATION between captures\n")

    # Live preview thread (all cameras quad view)
    preview_running = True

    def annotate_fixed_cam(bgr, cube_det, ci, K_c, D_c):
        """Annotate fixed camera with ArUco cube detection (cube IDs only)."""
        out = bgr.copy()
        corners_list, ids = cube_det.detect(out)

        # Filter to cube marker IDs only (0~4)
        cube_marker_ids = set(cube_cfg.marker_ids)
        if ids is not None and len(corners_list) > 0:
            filtered_corners = []
            filtered_ids = []
            for c, mid in zip(corners_list, ids):
                if int(mid) in cube_marker_ids:
                    filtered_corners.append(c)
                    filtered_ids.append(int(mid))
            corners_list = filtered_corners
            ids = np.array(filtered_ids) if filtered_ids else None

        n = 0 if ids is None else len(ids)

        if ids is not None and len(corners_list) > 0:
            try:
                draw_ids = ids.reshape(-1, 1) if getattr(ids, "ndim", 1) == 1 else ids
                cv2.aruco.drawDetectedMarkers(out, corners_list, draw_ids)
            except Exception:
                pass

            # Draw cube axes if PnP works
            pnp_ok, rvec, tvec, _, reproj = cube_det.solve_pnp_cube(
                bgr, K_c, D_c, use_ransac=False, min_markers=1,
                reproj_thr_mean_px=6.0, return_reproj=True,
            )
            if pnp_ok and rvec is not None:
                cv2.drawFrameAxes(out, K_c, D_c, rvec, tvec, 0.03)

        ids_txt = ",".join(str(int(x)) for x in ids) if ids is not None else "-"
        lines = [f"cam{ci} [FIXED]", f"cube={n}mkr"]
        y = 24
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (4, y - 18), (10 + tw, y + 4), (0, 0, 0), -1)
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y += 22
        return out

    def build_quad_preview():
        """Build 2x2 quad image from all cameras."""
        cam_order = sorted(cams.keys())
        tiles = []
        tile_h, tile_w = 480, 640

        for ci in cam_order:
            color, _, _ = cams[ci].get_latest()
            if color is None:
                blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(blank, f"cam{ci} N/A", (20, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                tiles.append(blank)
                continue

            tile_h, tile_w = color.shape[:2]

            if ci == gi:
                ann = annotate_gripper(color, K, D, ci, charuco, cube)
                tiles.append(ann)
            elif cube is not None and ci in K_map:
                # Fixed camera: ArUco cube detection
                ann = annotate_fixed_cam(color, cube, ci, K_map[ci], D_map[ci])
                tiles.append(ann)
            else:
                tiles.append(color)

        # Pad to 4 tiles
        while len(tiles) < 4:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        tiles = tiles[:4]

        top = cv2.hconcat([tiles[0], tiles[1]])
        bottom = cv2.hconcat([tiles[2], tiles[3]])
        return cv2.vconcat([top, bottom])

    def preview_loop():
        while preview_running:
            try:
                quad = build_quad_preview()
                ph = int(quad.shape[0] * 0.5)
                pw = int(quad.shape[1] * 0.5)
                preview = cv2.resize(quad, (pw, ph))
                cv2.imshow("Live Preview (all cameras)", preview)
                key = cv2.waitKey(100) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            except Exception as e:
                print(f"[ERROR] preview_loop crashed: {e}")
                time.sleep(0.2)

    if args.show:
        t = threading.Thread(target=preview_loop, daemon=True)
        t.start()
        print("[INFO] Live quad preview started (gripper: ChArUco, fixed: ArUco cube)")

    # Main capture loop
    try:
        while True:
            data = sock.recv(8192)
            if not data:
                print("[INFO] Server disconnected.")
                break

            msg = json.loads(data.decode("utf-8").strip())
            cmd = msg.get("command", "")

            if cmd == "quit":
                print("[INFO] Server sent quit.")
                break

            if cmd == "capture":
                capture_tcp = msg.get("capture_pose_6dof")
                pose_idx = msg.get("pose_index", event_id)

                print(f"\n[Capture] Signal received (pose={pose_idx})")
                if capture_tcp:
                    print(f"  TCP: [{capture_tcp[0]:.1f}, {capture_tcp[1]:.1f}, {capture_tcp[2]:.1f}, "
                          f"{capture_tcp[3]:.1f}, {capture_tcp[4]:.1f}, {capture_tcp[5]:.1f}]")

                # Wait for settle
                time.sleep(args.settle_time)

                # Capture and detect
                color, depth, ts_ms = cam.get_latest()
                if color is None:
                    print("[SKIP] No image from camera")
                    resp = json.dumps({"action": "captured", "status": "skipped"})
                    sock.sendall(resp.encode("utf-8"))
                    continue

                # --- ChArUco detection (gripper camera) ---
                ch_corners, ch_ids, n_corners, _, _ = charuco.detect(color)
                ok, rvec, tvec = False, None, None
                reproj = None
                if ch_corners is not None and n_corners >= 4:
                    ok, rvec, tvec, n_corners, reproj = safe_estimate_charuco(charuco, color, K, D)

                if not ok or n_corners < args.min_corners:
                    print(f"[SKIP] Not enough corners: {n_corners} (min={args.min_corners})")
                    resp = json.dumps({"action": "captured", "status": "skipped"})
                    sock.sendall(resp.encode("utf-8"))
                    continue

                # --- ArUco cube detection (gripper camera) ---
                cu_corners, cu_ids = cube.detect(color)
                cu_ok, cu_rvec, cu_tvec, cu_reproj = False, None, None, None
                n_cube = 0 if cu_ids is None else len(cu_ids)
                if cu_ids is not None and n_cube >= 1:
                    pnp_ok, rv, tv, used_ids, c_reproj = cube.solve_pnp_cube(
                        color, K, D, use_ransac=False, min_markers=1,
                        reproj_thr_mean_px=6.0, return_reproj=True,
                    )
                    if pnp_ok:
                        cu_ok, cu_rvec, cu_tvec, cu_reproj = True, rv, tv, c_reproj

                # Save image
                fid = int(event_id)
                rgb_rel = f"cam{gi}/rgb_{fid:05d}.jpg"
                cv2.imwrite(os.path.join(root, rgb_rel), color)

                depth_rel = None
                if args.save_depth and depth is not None:
                    depth_rel = f"cam{gi}/depth_{fid:05d}.png"
                    cv2.imwrite(os.path.join(root, depth_rel), depth)

                # Save annotated image
                charuco_result = (ok, rvec, tvec, n_corners, reproj, ch_corners, ch_ids)
                cube_result = (cu_ok, cu_rvec, cu_tvec, cu_corners, cu_ids)
                ann = annotate_gripper(color, K, D, gi, charuco, cube,
                                       charuco_result=charuco_result, cube_result=cube_result)
                ann_rel = f"cam{gi}/annotated_{fid:05d}.jpg"
                cv2.imwrite(os.path.join(root, ann_rel), ann)

                # Build T_cam_board
                T_cam_board = rodrigues_to_Rt(rvec, tvec)

                # Build T_base_gripper
                T_base_gripper = None
                if capture_tcp:
                    T_base_gripper = euler_deg_to_matrix(*capture_tcp)

                # Record
                gripper_rec = {
                    "rgb_path": rgb_rel,
                    "depth_path": depth_rel,
                    "annotated_path": ann_rel,
                    "charuco": {
                        "n_corners": n_corners,
                        "reproj_error_px": reproj,
                        "rvec": rvec.flatten().tolist(),
                        "tvec": tvec.flatten().tolist(),
                        "T_cam_board_4x4": T_cam_board.tolist(),
                    },
                }

                # Gripper cube detection result
                if cu_ok and cu_rvec is not None:
                    T_cam_cube = rodrigues_to_Rt(cu_rvec, cu_tvec)
                    gripper_rec["cube_pnp"] = {
                        "ok": True,
                        "n_markers": n_cube,
                        "marker_ids": [int(x) for x in np.array(cu_ids).flatten()],
                        "reproj_mean_px": cu_reproj["err_mean"] if cu_reproj else None,
                        "rvec": cu_rvec.flatten().tolist(),
                        "tvec": cu_tvec.flatten().tolist(),
                        "T_cam_cube_4x4": T_cam_cube.tolist(),
                    }

                cap_rec = {
                    "event_id": fid,
                    "pose_index": pose_idx,
                    "saved": True,
                    "gripper": gripper_rec,
                    "fixed_cams": {},
                }

                if capture_tcp:
                    tcp_f = [float(x) for x in capture_tcp]
                    cap_rec["robot_pose_6dof"] = tcp_f
                    cap_rec["T_base_gripper_4x4"] = T_base_gripper.tolist()

                # --- Fixed cameras: detect ArUco cube ---
                if args.also_detect_cube:
                    for ci in fixed_cam_ids:
                        if ci not in cams or ci not in K_map:
                            continue
                        fc_color, fc_depth, _ = cams[ci].get_latest()
                        if fc_color is None:
                            continue

                        # Save fixed camera image
                        fc_rgb_rel = f"cam{ci}/rgb_{fid:05d}.jpg"
                        cv2.imwrite(os.path.join(root, fc_rgb_rel), fc_color)

                        fc_depth_rel = None
                        if args.save_depth and fc_depth is not None:
                            fc_depth_rel = f"cam{ci}/depth_{fid:05d}.png"
                            cv2.imwrite(os.path.join(root, fc_depth_rel), fc_depth)

                        # Detect cube
                        _, ids = cube.detect(fc_color)
                        n_mkr = 0 if ids is None else len(ids)

                        fc_rec = {
                            "rgb_path": fc_rgb_rel,
                            "depth_path": fc_depth_rel,
                            "n_markers": n_mkr,
                            "marker_ids": [] if ids is None else [int(x) for x in np.array(ids).flatten()],
                        }

                        # Cube PnP
                        if n_mkr >= args.min_markers:
                            pnp_ok, c_rvec, c_tvec, used_ids, c_reproj = cube.solve_pnp_cube(
                                fc_color, K_map[ci], D_map[ci],
                                use_ransac=False, min_markers=1,
                                reproj_thr_mean_px=6.0, return_reproj=True,
                            )
                            if pnp_ok and c_reproj:
                                T_cam_cube = rodrigues_to_Rt(c_rvec, c_tvec)
                                fc_rec["cube_pnp"] = {
                                    "ok": True,
                                    "used_ids": [int(x) for x in np.array(used_ids).flatten()],
                                    "reproj_mean_px": c_reproj["err_mean"],
                                    "rvec": c_rvec.flatten().tolist(),
                                    "tvec": c_tvec.flatten().tolist(),
                                    "T_cam_cube_4x4": T_cam_cube.tolist(),
                                }

                        cap_rec["fixed_cams"][str(ci)] = fc_rec

                meta["captures"].append(cap_rec)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                # Print summary
                fixed_summary = ""
                if cap_rec["fixed_cams"]:
                    parts = []
                    for ci_str, fc in cap_rec["fixed_cams"].items():
                        n = fc["n_markers"]
                        has_pnp = "cube_pnp" in fc
                        parts.append(f"cam{ci_str}:{n}mkr{'✓' if has_pnp else ''}")
                    fixed_summary = " | fixed: " + " ".join(parts)

                reproj_txt = f"reproj={reproj:.3f}px" if reproj is not None else "reproj=N/A"
                cube_txt = f"cube={n_cube}mkr" + ("✓" if cu_ok else "")
                print(f"[SAVE] event={fid} corners={n_corners} {reproj_txt} {cube_txt}{fixed_summary}")
                event_id += 1

                resp = json.dumps({"action": "captured", "status": "success"})
                sock.sendall(resp.encode("utf-8"))

            elif cmd == "detect":
                # Visual servoing: detect cube from gripper camera
                color_det, _, _ = cam.get_latest()
                if color_det is None:
                    resp = json.dumps({"ok": False, "reason": "no_image"})
                    sock.sendall(resp.encode("utf-8"))
                    continue

                det_ok, det_rv, det_tv, det_used = cube.solve_pnp_cube(
                    color_det, K, D, use_ransac=False, min_markers=1,
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
                sock.sendall(resp.encode("utf-8"))

    finally:
        preview_running = False
        sock.close()
        for c in cams.values():
            c.stop()
        cv2.destroyAllWindows()

    print(f"\n[DONE] ChArUco captures: {event_id}")
    print(f"  Meta saved: {meta_path}")

    if event_id >= 5:
        print(f"\n  Next step:")
        print(f"    python Step3_in_calibration.py \\")
        print(f"      --charuco_folder {root} \\")
        print(f"      --intrinsics_dir {args.intrinsics_dir} \\")
        print(f"      --gripper_cam_idx {gi}")
    else:
        print(f"\n  [WARN] Need at least 5 captures (have {event_id}). Add more with rotation diversity.")


if __name__ == "__main__":
    main()
