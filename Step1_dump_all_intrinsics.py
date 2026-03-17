# Step1_dump_all_intrinsics.py

"""
Step 1: 연결된 모든 RealSense 카메라 intrinsics 저장
 - Saves per-camera npz (K, D, depth_scale, etc.)
 - Saves device_map.json with serial -> cam_idx mapping
 * 그리퍼 카메라 사용시 : User must label which cam_idx is the gripper camera

명령어:
  python Step1_dump_all_intrinsics.py \
  --out_dir ./intrinsics \
  --color_w 640 \
  --color_h 480 \
  --fps 15

결과물:
  intrinsics/
    device_map.json
    cam0.npz, cam1.npz, cam2.npz, cam3.npz, cam4.npz
    depth_scales.json
    intrinsics_by_serial/
"""

import os
import json
import time
import argparse
import numpy as np
import pyrealsense2 as rs

# ****** K,D = intrinsics -> camera matrix, distortion coeffs
def _intr_to_KD(intr: rs.intrinsics):
    fx, fy = float(intr.fx), float(intr.fy)
    cx, cy = float(intr.ppx), float(intr.ppy)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)
    return K, D

def main():
    parser = argparse.ArgumentParser(description="Dump intrinsics for all RealSense cameras")
    parser.add_argument("--out_dir", type=str, default="intrinsics")
    parser.add_argument("--color_w", type=int, default=640)
    parser.add_argument("--color_h", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--gripper_serial", type=str, default=None,
                        help="Serial number of the gripper camera (optional, for labeling)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    by_serial_dir = os.path.join(out_dir, "intrinsics_by_serial")
    os.makedirs(by_serial_dir, exist_ok=True)

    map_path = os.path.join(out_dir, "device_map.json")
    scales_path = os.path.join(out_dir, "depth_scales.json")

    # ****** 장치 검색 및 매핑
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[ERROR] No RealSense devices found.")
        return

    detected = []
    for d in devs:
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        detected.append({"serial": serial, "name": name})

    detected_serials = [x["serial"] for x in detected]
    print(f"[INFO] Found {len(detected)} RealSense devices:")
    for x in detected:
        print(f"  serial={x['serial']}  name={x['name']}")

    # ****** device map 생성/업데이트 (serial -> cam_idx)
    existing_map = None
    if os.path.exists(map_path):
        with open(map_path, "r") as f:
            existing_map = json.load(f)

    if existing_map is not None:
        serial_to_idx = dict(existing_map.get("serial_to_idx", {}))
        next_idx = 0 if len(serial_to_idx) == 0 else (max(serial_to_idx.values()) + 1)
        for s in detected_serials:
            if s not in serial_to_idx:
                serial_to_idx[s] = next_idx
                next_idx += 1
        print("[INFO] Updated existing device_map.json")
    else:
        detected_serials_sorted = sorted(detected_serials)
        serial_to_idx = {s: i for i, s in enumerate(detected_serials_sorted)}
        print("[INFO] Created new device_map.json (sorted by serial)")

    # ****** 그리퍼카메라 인덱스 설정 (사용자 입력 또는 기존 맵에서 유지)
    gripper_cam_idx = None
    if args.gripper_serial and args.gripper_serial in serial_to_idx:
        gripper_cam_idx = serial_to_idx[args.gripper_serial]

    map_obj = {
        "created_at_epoch": existing_map.get("created_at_epoch", time.time()) if existing_map else time.time(),
        "updated_at_epoch": time.time(),
        "serial_to_idx": serial_to_idx,
        "gripper_cam_idx": gripper_cam_idx,
        "gripper_serial": args.gripper_serial,
        "detected_now": detected,
    }
    with open(map_path, "w") as f:
        json.dump(map_obj, f, indent=2)
    print(f"[SAVE] {map_path}")

    # ****** 각 장치의 intrinsics 읽어서 저장
    depth_scales = {"updated_at_epoch": time.time(), "serial_to_depth_scale_m_per_unit": {}}
    idx_serial_pairs = sorted([(serial_to_idx[s], s) for s in detected_serials], key=lambda x: x[0])

    print("\n[INFO] Camera index assignment:")
    for idx, s in idx_serial_pairs:
        tag = " (GRIPPER)" if idx == gripper_cam_idx else " (FIXED)"
        print(f"  cam{idx}: serial={s}{tag}")

    for cam_idx, serial in idx_serial_pairs:
        dev = None
        for d in devs:
            if d.get_info(rs.camera_info.serial_number) == serial:
                dev = d
                break
        if dev is None:
            continue

        # Depth scale
        try:
            ds = float(dev.first_depth_sensor().get_depth_scale())
            depth_scales["serial_to_depth_scale_m_per_unit"][serial] = ds
        except Exception as e:
            print(f"[WARN] depth_scale read failed for {serial}: {e}")
            ds = None

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, args.color_w, args.color_h, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.color_w, args.color_h, rs.format.z16, args.fps)

        try:
            profile = pipeline.start(config)

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            Kc, Dc = _intr_to_KD(color_stream.get_intrinsics())

            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            Kd, Dd = _intr_to_KD(depth_stream.get_intrinsics())

            try:
                extr = depth_stream.get_extrinsics_to(color_stream)
                R_dc = np.array(extr.rotation, dtype=np.float64).reshape(3, 3)
                t_dc = np.array(extr.translation, dtype=np.float64).reshape(3, 1)
            except Exception:
                R_dc = np.eye(3, dtype=np.float64)
                t_dc = np.zeros((3, 1), dtype=np.float64)

            is_gripper = (cam_idx == gripper_cam_idx)

            npz_path = os.path.join(out_dir, f"cam{cam_idx}.npz")
            np.savez(npz_path,
                     serial=serial,
                     is_gripper=is_gripper,
                     color_K=Kc, color_D=Dc,
                     depth_K=Kd, depth_D=Dd,
                     depth_scale_m_per_unit=(ds if ds is not None else np.nan),
                     color_w=args.color_w, color_h=args.color_h, fps=args.fps,
                     R_depth_to_color=R_dc, t_depth_to_color=t_dc)
            print(f"[SAVE] {npz_path}")

            serial_npz = os.path.join(by_serial_dir, f"serial_{serial}.npz")
            np.savez(serial_npz, serial=serial, cam_idx=cam_idx,
                     color_K=Kc, color_D=Dc, depth_K=Kd, depth_D=Dd,
                     depth_scale_m_per_unit=(ds if ds is not None else np.nan))
            print(f"[SAVE] {serial_npz}")

            tag = "GRIPPER" if is_gripper else "FIXED"
            print(f"[INFO] cam{cam_idx} ({tag}) serial={serial}")
            print(f"       color K:\n{Kc}")
            print(f"       depth_scale = {ds}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    with open(scales_path, "w") as f:
        json.dump(depth_scales, f, indent=2)
    print(f"\n[SAVE] {scales_path}")
    print("[DONE] Step1_dump_all_intrinsics.py complete.")


if __name__ == "__main__":
    main()
