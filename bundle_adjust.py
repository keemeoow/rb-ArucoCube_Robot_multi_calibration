#!/usr/bin/env python3
"""Bundle adjustment refinement of an existing calibration.

Loads a calib_out/ directory produced by Step3, then jointly optimizes:
  - T_base_C{ci} for each fixed camera
  - T_gripper_cam (hand-eye)
  - Per-set T_base_O (cube pose, one per set)

Cost function: reprojection error of all visible cube marker corners across
all (camera, event, marker) observations.

Usage:
  python3 bundle_adjust.py \
      --root_folder ./<session> \
      --intrinsics_dir ./intrinsics \
      --calib_dir ./<session>/calib_out
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aruco_cube import ArucoCubeModel, inv_T
from config import CubeConfig
from calibration_runtime_utils import (
    load_intrinsics_color,
    load_robot_pose_from_capture,
    get_capture_set_index,
)


# ─────────────────────── SE(3) parameterization ───────────────────────

def T_from_vec(v: np.ndarray) -> np.ndarray:
    """6-dof vector (rotvec[3], trans[3]) → 4×4 SE(3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    T[:3, 3] = v[3:6]
    return T


def vec_from_T(T: np.ndarray) -> np.ndarray:
    """4×4 SE(3) → 6-dof vector."""
    rot = Rotation.from_matrix(np.asarray(T[:3, :3], dtype=np.float64)).as_rotvec()
    trans = np.asarray(T[:3, 3], dtype=np.float64)
    return np.concatenate([rot, trans])


# ─────────────────────── Observation collection ───────────────────────

def collect_observations(meta, root_folder, intrinsics_dir, all_cam_ids,
                         gripper_cam_idx, cube_cfg, min_markers_per_cam=2):
    """Build observation list for bundle adjustment.

    Only includes (cam, event) pairs that detected at least `min_markers_per_cam`
    cube markers — single-marker views have IPPE ambiguity that BA can't resolve
    cleanly, so we skip them.

    Returns list of dicts: {cam_idx, event_id, set_index, marker_id, T_B_G,
                            corners_2d (4,2), obj_corners (4,3 in cube frame), K, D}.
    """
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics_color(intrinsics_dir, ci)

    model = ArucoCubeModel(cube_cfg)

    obs = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        sidx = get_capture_set_index(cap)
        if sidx is None:
            continue
        T_B_G = load_robot_pose_from_capture(cap)
        if T_B_G is None:
            continue
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in K_map or not cinfo.get("saved"):
                continue
            valid_markers = [m for m in cinfo.get("markers", [])
                             if int(m.get("marker_id", -1)) in cube_cfg.id_to_face
                             and np.asarray(m.get("corners_2d", []), dtype=np.float64).shape == (4, 2)]
            if len(valid_markers) < int(min_markers_per_cam):
                continue
            for m in valid_markers:
                mid = int(m.get("marker_id", -1))
                corners = np.asarray(m["corners_2d"], dtype=np.float64)
                # Apply corner reorder (so detected order matches model)
                reorder = cube_cfg.corner_reorder.get(mid, [0, 1, 2, 3])
                corners_reordered = corners[reorder]
                obj_corners = model.marker_corners_in_rig(mid)  # (4, 3) in cube frame
                obs.append({
                    "cam_idx": ci,
                    "event_id": eid,
                    "set_index": int(sidx),
                    "marker_id": mid,
                    "is_gripper": (gripper_cam_idx is not None and ci == gripper_cam_idx),
                    "T_B_G": T_B_G,
                    "corners_2d": corners_reordered,
                    "obj_corners": obj_corners,
                    "K": K_map[ci],
                    "D": D_map[ci],
                })
    return obs


# ─────────────────────── Parameter packing ───────────────────────

class ParamLayout:
    """Maps {fixed_cam, T_gripper_cam, T_base_O_set} ↔ flat params vector."""
    def __init__(self, fixed_cam_ids, set_indices, gripper_present):
        self.fixed_cam_ids = sorted(int(c) for c in fixed_cam_ids)
        self.set_indices = sorted(int(s) for s in set_indices)
        self.gripper_present = bool(gripper_present)
        self.idx = {}
        cur = 0
        for ci in self.fixed_cam_ids:
            self.idx[("cam", ci)] = (cur, cur + 6)
            cur += 6
        if gripper_present:
            self.idx[("gripper",)] = (cur, cur + 6)
            cur += 6
        for sidx in self.set_indices:
            self.idx[("set", sidx)] = (cur, cur + 6)
            cur += 6
        self.n_params = cur

    def pack(self, T_base_Ci, T_gripper_cam, T_base_O_by_set):
        v = np.zeros(self.n_params, dtype=np.float64)
        for ci in self.fixed_cam_ids:
            i0, i1 = self.idx[("cam", ci)]
            v[i0:i1] = vec_from_T(T_base_Ci[ci])
        if self.gripper_present:
            i0, i1 = self.idx[("gripper",)]
            v[i0:i1] = vec_from_T(T_gripper_cam)
        for sidx in self.set_indices:
            i0, i1 = self.idx[("set", sidx)]
            v[i0:i1] = vec_from_T(T_base_O_by_set[sidx])
        return v

    def unpack(self, v):
        T_base_Ci = {}
        for ci in self.fixed_cam_ids:
            i0, i1 = self.idx[("cam", ci)]
            T_base_Ci[ci] = T_from_vec(v[i0:i1])
        T_gripper_cam = None
        if self.gripper_present:
            i0, i1 = self.idx[("gripper",)]
            T_gripper_cam = T_from_vec(v[i0:i1])
        T_base_O_by_set = {}
        for sidx in self.set_indices:
            i0, i1 = self.idx[("set", sidx)]
            T_base_O_by_set[sidx] = T_from_vec(v[i0:i1])
        return T_base_Ci, T_gripper_cam, T_base_O_by_set


# ─────────────────────── Cost function ───────────────────────

def reprojection_residuals(params, layout, observations):
    """Flat residual vector (in pixels) for least_squares."""
    T_base_Ci, T_gripper_cam, T_base_O_by_set = layout.unpack(params)
    residuals = []
    for ob in observations:
        ci = ob["cam_idx"]
        sidx = ob["set_index"]
        if sidx not in T_base_O_by_set:
            continue
        T_base_O = T_base_O_by_set[sidx]
        if ob["is_gripper"]:
            if T_gripper_cam is None:
                continue
            T_base_cam = ob["T_B_G"] @ T_gripper_cam
        else:
            if ci not in T_base_Ci:
                continue
            T_base_cam = T_base_Ci[ci]
        # T_C_O = inv(T_base_cam) @ T_base_O
        T_C_O = inv_T(T_base_cam) @ T_base_O
        rvec, _ = cv2.Rodrigues(T_C_O[:3, :3])
        tvec = T_C_O[:3, 3].reshape(3, 1)
        proj, _ = cv2.projectPoints(
            ob["obj_corners"].reshape(-1, 1, 3),
            rvec, tvec, ob["K"], ob["D"],
        )
        proj = proj.reshape(-1, 2)
        residuals.append((proj - ob["corners_2d"]).reshape(-1))
    if not residuals:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(residuals)


# ─────────────────────── Driver ───────────────────────

def load_initial_calibration(calib_dir, fixed_cam_ids, set_indices, gripper_present):
    calib_dir = Path(calib_dir)
    T_base_Ci = {}
    for ci in fixed_cam_ids:
        path = calib_dir / f"T_base_C{int(ci)}.npy"
        if path.exists():
            T_base_Ci[int(ci)] = np.load(str(path))
    T_gripper_cam = None
    if gripper_present:
        path = calib_dir / "T_gripper_cam.npy"
        if path.exists():
            T_gripper_cam = np.load(str(path))
    T_base_O_by_set = {}
    runtime = calib_dir / "internal_runtime"
    if runtime.exists():
        for sidx in set_indices:
            path = runtime / f"T_base_O_set{int(sidx)}.npy"
            if path.exists():
                T_base_O_by_set[int(sidx)] = np.load(str(path))
    return T_base_Ci, T_gripper_cam, T_base_O_by_set


def run_bundle_adjust(root_folder, intrinsics_dir, calib_dir,
                      max_nfev=200, loss="huber", f_scale=1.0,
                      verbose=True):
    root = Path(root_folder).resolve()
    intr = Path(intrinsics_dir).resolve()
    calib = Path(calib_dir).resolve()

    with open(root / "meta.json") as f:
        meta = json.load(f)

    cube_cfg = CubeConfig()  # config_py canonical defaults

    # Discover cams + sets
    all_cam_ids = sorted({
        int(k) for cap in meta.get("captures", [])
        for k, v in cap.get("cams", {}).items() if v.get("saved")
    })
    set_indices = sorted({
        int(s) for cap in meta.get("captures", [])
        if (s := get_capture_set_index(cap)) is not None
    })
    gripper_cam_idx = meta.get("gripper_cam_idx")
    if gripper_cam_idx is None:
        gripper_cam_idx = max(all_cam_ids)
    fixed_cam_ids = [c for c in all_cam_ids if c != int(gripper_cam_idx)]

    if verbose:
        print(f"[BA] cams={all_cam_ids} fixed={fixed_cam_ids} gripper={gripper_cam_idx} sets={set_indices}")

    T_base_Ci_init, T_gripper_cam_init, T_base_O_by_set_init = load_initial_calibration(
        calib, fixed_cam_ids, set_indices, gripper_present=True)

    if not T_base_Ci_init or T_gripper_cam_init is None or not T_base_O_by_set_init:
        print(f"[BA] missing initial transforms — skipping bundle adjustment")
        return None

    layout = ParamLayout(fixed_cam_ids, set_indices, gripper_present=True)
    x0 = layout.pack(T_base_Ci_init, T_gripper_cam_init, T_base_O_by_set_init)

    obs = collect_observations(meta, str(root), str(intr), all_cam_ids,
                               int(gripper_cam_idx), cube_cfg)
    if verbose:
        print(f"[BA] observations: {len(obs)} markers across all cam/event combos")

    if not obs:
        print(f"[BA] no observations — skipping")
        return None

    initial_residuals = reprojection_residuals(x0, layout, obs)
    rms_initial = float(np.sqrt(np.mean(initial_residuals ** 2)))
    if verbose:
        print(f"[BA] initial RMS reprojection: {rms_initial:.4f} px")

    t0 = time.time()
    result = least_squares(
        reprojection_residuals, x0, args=(layout, obs),
        method="trf",
        loss=loss,
        f_scale=float(f_scale),
        max_nfev=int(max_nfev),
        verbose=2 if verbose else 0,
        xtol=1e-9, ftol=1e-9, gtol=1e-9,
    )
    dt = time.time() - t0
    rms_final = float(np.sqrt(np.mean(result.fun ** 2)))
    if verbose:
        print(f"[BA] optimization done in {dt:.1f}s, nfev={result.nfev}, "
              f"final RMS: {rms_final:.4f} px ({(rms_initial - rms_final)/rms_initial * 100:.1f}% improvement)")

    T_base_Ci_ref, T_gripper_cam_ref, T_base_O_by_set_ref = layout.unpack(result.x)

    # Save refined transforms
    for ci, T in T_base_Ci_ref.items():
        np.save(str(calib / f"T_base_C{int(ci)}.npy"), np.asarray(T, dtype=np.float64))
    np.save(str(calib / "T_gripper_cam.npy"), np.asarray(T_gripper_cam_ref, dtype=np.float64))
    # Update T_base_O.npy with average across sets
    if len(T_base_O_by_set_ref) > 1:
        ts = np.array([T_base_O_by_set_ref[s][:3, 3] for s in T_base_O_by_set_ref])
        Rs = np.array([T_base_O_by_set_ref[s][:3, :3] for s in T_base_O_by_set_ref])
        T_avg = np.eye(4, dtype=np.float64)
        T_avg[:3, 3] = ts.mean(axis=0)
        R_mean = Rs.mean(axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        T_avg[:3, :3] = U @ Vt
        np.save(str(calib / "T_base_O.npy"), T_avg)
    else:
        np.save(str(calib / "T_base_O.npy"), next(iter(T_base_O_by_set_ref.values())))
    runtime = calib / "internal_runtime"
    runtime.mkdir(exist_ok=True)
    for sidx, T in T_base_O_by_set_ref.items():
        np.save(str(runtime / f"T_base_O_set{int(sidx)}.npy"), np.asarray(T, dtype=np.float64))

    # Save BA report
    ba_report = {
        "n_observations": int(len(obs)),
        "n_params": int(layout.n_params),
        "n_iterations": int(result.nfev),
        "rms_initial_px": rms_initial,
        "rms_final_px": rms_final,
        "improvement_pct": float((rms_initial - rms_final) / rms_initial * 100.0),
        "duration_s": float(dt),
        "loss": str(loss),
        "f_scale": float(f_scale),
        "status": int(result.status),
        "message": str(result.message),
    }
    with open(calib / "bundle_adjust_report.json", "w") as f:
        json.dump(ba_report, f, indent=2)
    if verbose:
        print(f"[BA] report saved: {calib / 'bundle_adjust_report.json'}")

    return ba_report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_folder", required=True)
    ap.add_argument("--intrinsics_dir", required=True)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--max_nfev", type=int, default=200)
    ap.add_argument("--loss", default="huber", choices=["linear", "soft_l1", "huber", "cauchy"])
    ap.add_argument("--f_scale", type=float, default=1.0,
                    help="Robust loss scale (pixels). Below this: ~quadratic; above: down-weighted.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run_bundle_adjust(args.root_folder, args.intrinsics_dir, args.calib_dir,
                      max_nfev=args.max_nfev, loss=args.loss, f_scale=args.f_scale,
                      verbose=not args.quiet)


if __name__ == "__main__":
    main()
