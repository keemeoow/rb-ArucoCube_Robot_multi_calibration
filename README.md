# Unified Multi-Fixed-Camera + Robot Hand-Eye + ArUco Cube Calibration

## Overview

This pipeline calibrates a system consisting of:
- **N fixed cameras** (RealSense, mounted around workspace)
- **1 robot gripper camera** (RealSense, eye-in-hand)
- **ArUco cube** (calibration target)

### Coordinate Frames
```
Base (Robot)
 ├── Gripper (EE) ── gTc ──> Gripper Camera
 │                              │
 │                         sees ArUco Cube
 │                              │
 ├── bTo (cube in base) <──────-┘
 │
 └── bTfi (fixed cam i in base)
      ├── Fixed Cam 0 (ref)
      ├── Fixed Cam 1
      ├── Fixed Cam 2
      └── Fixed Cam 3
```

### Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `Step1_dump_all_intrinsics.py` | Dump intrinsics for all cameras (fixed + gripper) |
| 2 | `Step2_capture_cube_poses.py` | Capture ArUco cube from all cameras + record robot joints |
| 3 | `Step3_calibrate_all.py` | Compute: fixed cam extrinsics, hand-eye (gTc), fixed-to-base transforms |
| 4 | `Step4_verify_and_fuse.py` | Verify calibration accuracy, optionally fuse point clouds |

### Usage
```bash
# Step 1: Dump intrinsics (connect ALL cameras)
python Step1_dump_all_intrinsics.py --out_dir ./intrinsics

# Step 2: Capture (robot moves cube, all cameras see it)
python Step2_capture_cube_poses.py \
  --root_folder ./data/session_01 \
  --intrinsics_dir ./intrinsics \
  --robot_ip 192.168.0.23 --robot_port 12348 \
  --save_depth --show

# Step 3: Calibrate everything
python Step3_calibrate_all.py \
  --root_folder ./data/session_01 \
  --intrinsics_dir ./intrinsics \
  --gripper_cam_idx 0 \
  --ref_fixed_cam_idx 1

# Step 4: Verify
python Step4_verify_and_fuse.py \
  --root_folder ./data/session_01 \
  --intrinsics_dir ./intrinsics
```
