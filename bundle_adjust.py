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


def _run_single_ba(x0, layout, obs, loss, f_scale, max_nfev, frozen_mask=None,
                   verbose=False):
    """단일 BA 실행. frozen_mask: True인 위치 파라미터를 잠금(잔차 미반영).
    least_squares는 직접 잠금을 지원 안 하므로, 잠긴 파라미터는 x0 그대로 두고
    내부적으로 미니멀하게 wrap.
    """
    if frozen_mask is None:
        return least_squares(
            reprojection_residuals, x0, args=(layout, obs),
            method="trf", loss=loss, f_scale=float(f_scale),
            max_nfev=int(max_nfev),
            verbose=2 if verbose else 0,
            xtol=1e-9, ftol=1e-9, gtol=1e-9)
    free_idx = np.where(~frozen_mask)[0]
    x_free0 = x0[free_idx].copy()

    def wrapper(xf):
        x = x0.copy()
        x[free_idx] = xf
        return reprojection_residuals(x, layout, obs)

    res = least_squares(
        wrapper, x_free0, method="trf", loss=loss, f_scale=float(f_scale),
        max_nfev=int(max_nfev), verbose=2 if verbose else 0,
        xtol=1e-9, ftol=1e-9, gtol=1e-9)
    # 결과를 full x로 패킹
    x_full = x0.copy()
    x_full[free_idx] = res.x
    # res 객체에 x_full 부착
    res.x = x_full
    res.fun = reprojection_residuals(x_full, layout, obs)
    return res


def _per_obs_rms(params, layout, observations):
    """Compute per-observation RMS reprojection error (one float per obs).

    Useful for outlier filtering between staged BA passes.
    """
    T_base_Ci, T_gripper_cam, T_base_O_by_set = layout.unpack(params)
    out = np.full(len(observations), np.inf, dtype=np.float64)
    for k, ob in enumerate(observations):
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
        T_C_O = inv_T(T_base_cam) @ T_base_O
        rvec, _ = cv2.Rodrigues(T_C_O[:3, :3])
        tvec = T_C_O[:3, 3].reshape(3, 1)
        proj, _ = cv2.projectPoints(
            ob["obj_corners"].reshape(-1, 1, 3),
            rvec, tvec, ob["K"], ob["D"],
        )
        proj = proj.reshape(-1, 2)
        diff = (proj - ob["corners_2d"]).reshape(-1)
        out[k] = float(np.sqrt(np.mean(diff ** 2)))
    return out


def _outlier_filter(obs, per_obs_rms, k_mad=3.0):
    """Drop observations whose per-marker RMS > median + k * MAD.

    Returns (kept_obs, n_dropped). MAD ≈ 1.4826*median(|x - med|).
    Per-camera bucket so cams with poor intrinsics aren't all dropped.
    """
    if not obs:
        return obs, 0
    cam_groups = {}
    for i, ob in enumerate(obs):
        cam_groups.setdefault(int(ob["cam_idx"]), []).append(i)
    keep = np.ones(len(obs), dtype=bool)
    for ci, idxs in cam_groups.items():
        vals = per_obs_rms[idxs]
        if not np.isfinite(vals).any():
            continue
        med = float(np.median(vals[np.isfinite(vals)]))
        mad = 1.4826 * float(np.median(np.abs(vals[np.isfinite(vals)] - med)))
        if mad < 1e-6:
            continue
        thr = med + k_mad * mad
        for i in idxs:
            if not np.isfinite(per_obs_rms[i]) or per_obs_rms[i] > thr:
                keep[i] = False
    kept = [ob for ok, ob in zip(keep, obs) if ok]
    n_dropped = int((~keep).sum())
    return kept, n_dropped


def _perturb_initial(x0, sigma_mm=10.0, sigma_deg=3.0, rng=None):
    """주어진 x0 (pack된 [tx,ty,tz,rx,ry,rz,...]+) 을 작게 흔든 시드 생성."""
    rng = rng or np.random.default_rng()
    x = x0.copy()
    # x layout: 매 6개씩 (tx,ty,tz,rx,ry,rz) 형태
    n_blocks = len(x) // 6
    sigma_t = sigma_mm / 1000.0  # m
    sigma_r = np.deg2rad(sigma_deg)
    for i in range(n_blocks):
        x[6 * i:6 * i + 3] += rng.normal(0, sigma_t, size=3)
        x[6 * i + 3:6 * i + 6] += rng.normal(0, sigma_r, size=3)
    return x


def run_bundle_adjust(root_folder, intrinsics_dir, calib_dir,
                      max_nfev=200, loss="huber", f_scale=1.0,
                      verbose=True,
                      multi_seed=5, staged=True):
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

    # Build slice masks once for staged freezing
    n_static = 6 * (len(fixed_cam_ids))
    n_he = 6 if T_gripper_cam_init is not None else 0

    def _mask(free_cams: bool, free_he: bool, free_sets: bool) -> np.ndarray:
        m = np.ones(layout.n_params, dtype=bool)  # True = frozen
        if free_cams:
            m[:n_static] = False
        if free_he and n_he:
            m[n_static:n_static + n_he] = False
        if free_sets:
            m[n_static + n_he:] = False
        return m

    stage_rms = {}
    n_dropped_total = 0

    if staged:
        # ── Stage 1: cube poses only — clean up rig with cams/HE pinned
        frozen_s1 = _mask(free_cams=False, free_he=False, free_sets=True)
        if verbose:
            print(f"[BA] stage-1: cube only (free={(~frozen_s1).sum()}/{layout.n_params})")
        res1 = _run_single_ba(x0, layout, obs, loss, f_scale, max_nfev // 2,
                              frozen_mask=frozen_s1, verbose=False)
        x_cur = res1.x.copy()
        rms_s1 = float(np.sqrt(np.mean(res1.fun ** 2)))
        stage_rms["stage1_cube_only"] = rms_s1
        if verbose:
            print(f"[BA] stage-1 RMS: {rms_s1:.4f} px")

        # ── Outlier filter: drop per-cam high-residual observations
        per_obs = _per_obs_rms(x_cur, layout, obs)
        obs_filt, n_dropped = _outlier_filter(obs, per_obs, k_mad=3.0)
        n_dropped_total = n_dropped
        if verbose:
            print(f"[BA] outlier filter: dropped {n_dropped}/{len(obs)} obs (k_mad=3)")
        if obs_filt:
            obs = obs_filt
            rms_after_filter = float(np.sqrt(np.mean(
                reprojection_residuals(x_cur, layout, obs) ** 2)))
            stage_rms["after_outlier_filter"] = rms_after_filter
            if verbose:
                print(f"[BA] RMS on kept obs: {rms_after_filter:.4f} px")

        # ── Stage 2a: fixed cams + cube free, HE frozen
        if n_static > 0:
            frozen_s2a = _mask(free_cams=True, free_he=False, free_sets=True)
            if verbose:
                print(f"[BA] stage-2a: fixed cams + cube (HE frozen, free={(~frozen_s2a).sum()})")
            res2a = _run_single_ba(x_cur, layout, obs, loss, f_scale, max_nfev,
                                   frozen_mask=frozen_s2a, verbose=False)
            x_cur = res2a.x.copy()
            rms_s2a = float(np.sqrt(np.mean(res2a.fun ** 2)))
            stage_rms["stage2a_fixedcams_he_frozen"] = rms_s2a
            if verbose:
                print(f"[BA] stage-2a RMS: {rms_s2a:.4f} px")

        # ── Stage 2b: HE + cube free, fixed cams frozen
        if n_he > 0:
            frozen_s2b = _mask(free_cams=False, free_he=True, free_sets=True)
            if verbose:
                print(f"[BA] stage-2b: HE + cube (fixed cams frozen, free={(~frozen_s2b).sum()})")
            res2b = _run_single_ba(x_cur, layout, obs, loss, f_scale, max_nfev,
                                   frozen_mask=frozen_s2b, verbose=False)
            x_cur = res2b.x.copy()
            rms_s2b = float(np.sqrt(np.mean(res2b.fun ** 2)))
            stage_rms["stage2b_he_cams_frozen"] = rms_s2b
            if verbose:
                print(f"[BA] stage-2b RMS: {rms_s2b:.4f} px")

        x_stage1 = x_cur
        rms_s1 = stage_rms.get("stage2b_he_cams_frozen",
                               stage_rms.get("stage2a_fixedcams_he_frozen", rms_s1))
    else:
        x_stage1 = x0
        rms_s1 = rms_initial

    # ── Stage 2: 전체 자유 + 다중 시드
    rng = np.random.default_rng(42)
    seeds = [x_stage1]
    for i in range(max(0, multi_seed - 1)):
        seeds.append(_perturb_initial(x_stage1, sigma_mm=8.0 * (i + 1),
                                       sigma_deg=2.0 * (i + 1), rng=rng))
    if verbose:
        print(f"[BA] stage-2: full BA with {len(seeds)} seeds")

    best_result = None
    best_rms = np.inf
    seed_summary = []
    for si, seed in enumerate(seeds):
        r = _run_single_ba(seed, layout, obs, loss, f_scale, max_nfev,
                            frozen_mask=None, verbose=False)
        rms_s = float(np.sqrt(np.mean(r.fun ** 2)))
        seed_summary.append({"seed_idx": si, "rms_px": rms_s,
                             "nfev": int(r.nfev), "status": int(r.status)})
        if verbose:
            print(f"   seed[{si}]: rms={rms_s:.4f}px  nfev={r.nfev}")
        if rms_s < best_rms:
            best_rms = rms_s
            best_result = r

    result = best_result
    dt = time.time() - t0
    rms_final = best_rms
    if verbose:
        print(f"[BA] optimization done in {dt:.1f}s, total seeds={len(seeds)}, "
              f"best RMS: {rms_final:.4f} px ({(rms_initial - rms_final)/rms_initial * 100:.1f}% improvement)")

    T_base_Ci_ref, T_gripper_cam_ref, T_base_O_by_set_ref = layout.unpack(result.x)

    # Safety guard: BA가 잔차를 악화시키면 초기 값을 유지 (Step3 결과 보호)
    if rms_final > rms_initial * 1.02:
        if verbose:
            print(f"[BA] WARNING: BA worsened RMS ({rms_initial:.4f} -> {rms_final:.4f}px). "
                  "Reverting to Step3 result.")
        T_base_Ci_ref = T_base_Ci_init
        T_gripper_cam_ref = T_gripper_cam_init
        T_base_O_by_set_ref = T_base_O_by_set_init
        rms_final = rms_initial

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
        "n_observations_dropped": int(n_dropped_total),
        "n_params": int(layout.n_params),
        "n_iterations": int(result.nfev),
        "rms_initial_px": rms_initial,
        "rms_stage1_px": float(rms_s1) if staged else None,
        "rms_final_px": rms_final,
        "stage_rms": stage_rms if staged else None,
        "improvement_pct": float((rms_initial - rms_final) / rms_initial * 100.0),
        "duration_s": float(dt),
        "loss": str(loss),
        "f_scale": float(f_scale),
        "status": int(result.status),
        "message": str(result.message),
        "multi_seed": int(multi_seed),
        "staged": bool(staged),
        "seed_summary": seed_summary if staged else None,
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
