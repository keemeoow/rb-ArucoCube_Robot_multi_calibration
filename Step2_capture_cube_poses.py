# Step2_capture_cube_poses.py
"""
Step 2: Capture ArUco cube images from ALL cameras
        while simultaneously recording robot TCP poses.

Two modes:
  A) Robot-controlled (--use_robot): Robot server sends 'capture' commands,
     client sends predefined 6D command, captures from all cameras.
  B) Manual (default): Press SPACE to capture when cube is visible.

For each capture event, saves:
  - RGB (+ optional depth) per camera
  - Robot TCP pose (if available)
  - Robot command joints/pose (if robot mode)

명령어 : 
[수동모드]
  python Step2_capture_cube_poses.py \
    --root_folder ./data/session_01 \
    --intrinsics_dir ./intrinsics \
    --save_depth --show

[로봇모드] - 이거로 사용!!
  python Step2_capture_cube_poses.py \
    --root_folder ./data/session_01 \
    --intrinsics_dir ./intrinsics \
    --use_robot \
    --robot_ip 192.168.0.23 \ --robot_port 12348 \
    --joint_file joints_handeye_calib.json \
    --settle_time 1.5 \
    --save_depth \
    --show \
    --min_cams_with_cube 2 \
    --min_markers 1
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional

import cv2

from camera import RealSenseCamera
from aruco_cube import ArucoCubeTarget
from config import CubeConfig
from robot_comm import RobotClient, euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    gripper_cam_idx = m.get("gripper_cam_idx", None)
    return serial_to_idx, gripper_cam_idx, map_path


def main():
    parser = argparse.ArgumentParser(description="Capture cube images from all cameras + robot poses")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)

    # Stream config
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    # Detection
    parser.add_argument("--min_markers", type=int, default=1,
                        help="Minimum markers visible per camera to accept capture")
    parser.add_argument("--min_cams_with_cube", type=int, default=2,
                        help="Minimum number of cameras that must see the cube")

    # Auto-save
    parser.add_argument("--auto_save", action="store_true")
    parser.add_argument("--stable_frames", type=int, default=3)
    parser.add_argument("--cooldown_ms", type=int, default=700)

    # Depth
    parser.add_argument("--save_depth", action="store_true")

    # Display
    parser.add_argument("--show", action="store_true")

    # Robot mode
    parser.add_argument("--use_robot", action="store_true",
                        help="Enable robot communication mode")
    parser.add_argument("--robot_ip", type=str, default="192.168.0.23")
    parser.add_argument("--robot_port", type=int, default=12348)
    parser.add_argument("--joint_file", type=str, default=None,
                        help="JSON file with list of 6D commands [[d1..d6], ...]")
    parser.add_argument("--settle_time", type=float, default=1.5,
                        help="Wait time (s) after robot moves before capturing")
    parser.add_argument("--query_robot_tcp", dest="query_robot_tcp", action="store_true",
                        help="Try querying Zeus for current TCP pose when capture command payload has no pose")
    parser.add_argument("--no_query_robot_tcp", dest="query_robot_tcp", action="store_false",
                        help="Disable extra TCP query after robot move")
    parser.add_argument("--allow_joint_as_pose_fallback", action="store_true",
                        help="If TCP pose is unavailable, store sent 6D command as robot_pose_6dof (권장X)")
    parser.set_defaults(query_robot_tcp=True)

    args = parser.parse_args()

    root = ensure_dir(args.root_folder)
    intr_dir = args.intrinsics_dir

    # ─── Load device map ───
    serial_to_idx, gripper_cam_idx, _ = load_device_map(intr_dir)
    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found.")

    if serial_to_idx is None:
        print("[WARN] No device_map.json. Run Step1 first. Using fallback.")
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

    print("[INFO] Cameras:")
    for idx, s in idx_serial_pairs:
        tag = "GRIPPER" if idx == gripper_cam_idx else "FIXED"
        print(f"  cam{idx}: {s} ({tag})")

    # ─── Start cameras ───
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

    # ─── Robot setup ───
    robot_client: Optional[RobotClient] = None
    joint_list: List[List[float]] = []

    if args.use_robot:
        if args.joint_file:
            with open(args.joint_file, "r") as f:
                joint_list = json.load(f)
            print(f"[INFO] Loaded {len(joint_list)} commands from {args.joint_file}")
        robot_client = RobotClient(args.robot_ip, args.robot_port)
        robot_client.connect()

    # ─── Meta ───
    meta = {
        "root_folder": os.path.abspath(root),
        "gripper_cam_idx": gripper_cam_idx,
        "captures": []
    }
    meta_path = os.path.join(root, "meta.json")

    event_id = 0
    stable_cnt = {ci: 0 for ci in cams}
    last_save_t = 0.0

    print("\nControls:")
    print("  SPACE : manual capture (all view check)")
    print("  ESC/q : quit\n")

    def do_capture(
        robot_pose_6dof: Optional[List[float]] = None,
        robot_joints_6: Optional[List[float]] = None,
        robot_pose_source: Optional[str] = None,
    ) -> bool:
        """Capture from all cameras. Returns True if successful."""
        nonlocal event_id, last_save_t

        frames: Dict[int, dict] = {}
        cams_with_cube = 0

        for ci, cam in cams.items():
            color, depth, ts_ms = cam.get_latest()
            if color is None:
                continue

            corners, ids = cube.detect(color)
            ok = (ids is not None) and (len(ids) >= args.min_markers)
            if ok:
                cams_with_cube += 1

            frames[ci] = {
                "color": color,
                "depth": depth,
                "ts_ms": ts_ms,
                "ok": ok,
                "ids": ([] if ids is None else [int(x) for x in ids]),
                "corners": corners,
                "ids_np": ids,
            }

        if cams_with_cube < args.min_cams_with_cube:
            print(f"[SKIP] Only {cams_with_cube}/{args.min_cams_with_cube} cams see cube.")
            return False

        fid = int(event_id)
        cap_rec = {"event_id": fid, "cams": {}}

        if robot_joints_6 is not None:
            cap_rec["robot_joints_6"] = [float(x) for x in robot_joints_6]

        if robot_pose_6dof is not None:
            pose6 = [float(x) for x in robot_pose_6dof]
            cap_rec["robot_pose_6dof"] = pose6
            cap_rec["robot_pose_source"] = str(robot_pose_source or "unknown")
            try:
                cap_rec["robot_pose_matrix_4x4"] = euler_deg_to_matrix(*pose6).tolist()
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

            cap_rec["cams"][str(ci)] = {
                "saved": True,
                "ts_ms": fr["ts_ms"],
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "ids": fr["ids"],
                "cube_visible": fr["ok"],
            }

        meta["captures"].append(cap_rec)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        visible_str = ", ".join([f"cam{ci}({'✓' if frames[ci]['ok'] else '✗'})" for ci in sorted(frames.keys())])
        src = cap_rec.get("robot_pose_source", "-")
        print(f"[SAVE] event_id={event_id} | {visible_str} | pose_src={src}")
        event_id += 1
        last_save_t = time.time() * 1000
        return True

    try:
        if args.use_robot and joint_list:
            # ─── Robot-controlled mode ───
            print("[MODE] Robot-controlled capture")
            for ji, joints in enumerate(joint_list):
                if all(float(x) == 0.0 for x in joints):
                    print("[INFO] End marker reached in joint list.")
                    break

                print(f"\n[Robot] Pose {ji+1}/{len(joint_list)}: {joints}")
                ok, tcp_pose, pose_source = robot_client.send_pose_and_wait_with_tcp(
                    joints=joints,
                    settle_time=args.settle_time,
                    query_tcp_if_missing=args.query_robot_tcp,
                )
                if not ok:
                    print("[INFO] Robot quit or error.")
                    break

                if tcp_pose is None and args.allow_joint_as_pose_fallback:
                    print("[WARN] TCP pose unavailable. Using command values as fallback robot_pose_6dof.")
                    tcp_pose = [float(x) for x in joints]
                    pose_source = "joint_fallback"
                elif tcp_pose is None:
                    print("[WARN] TCP pose unavailable. This capture will be saved without robot_pose_6dof.")

                do_capture(
                    robot_pose_6dof=tcp_pose,
                    robot_joints_6=[float(x) for x in joints],
                    robot_pose_source=pose_source,
                )

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
                    if ok:
                        stable_cnt[ci] += 1
                    else:
                        stable_cnt[ci] = 0

                    frames_view[ci] = {"color": color, "ok": ok, "corners": corners, "ids_np": ids}

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
                        txt = f"cam{ci}({tag}) ok={frames_view[ci]['ok']} stable={stable_cnt[ci]}"
                        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow(f"cam{ci}", img)

                key = cv2.waitKey(1) & 0xFF
                now_ms = time.time() * 1000.0

                if key == 27 or key == ord('q'):
                    break

                manual_trigger = (key == 32)  # SPACE

                if args.auto_save:
                    all_stable = all(stable_cnt.get(ci, 0) >= args.stable_frames for ci in cams)
                    if all_stable and (now_ms - last_save_t) >= args.cooldown_ms:
                        manual_trigger = True

                if manual_trigger:
                    do_capture(robot_pose_6dof=None)

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
