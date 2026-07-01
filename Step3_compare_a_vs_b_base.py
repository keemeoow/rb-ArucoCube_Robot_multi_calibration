#!/usr/bin/env python3
"""Compare camera<->robot-base registration two ways, on ONE capture session.

(a) set_cube_center-dependent  : base frame from the robot's reported cube pose
                                 (set_cube_center), anchored by cube-center
                                 POSITIONS (translation-only Kabsch).  Uses the
                                 A_placement frames (cube released on table).
(b) independent (eye-to-hand)  : base frame from robot FK + fixed-camera cube
                                 observations while the cube is RIGIDLY GRIPPED
                                 and the robot sweeps.  Uses the B_eyetohand frames.

Both yield T_base_Ci for the fixed cameras; the script reports each method's
self-consistency (Kabsch residual / hand-eye residual) and the per-camera
difference between (a) and (b).

Frame split is driven by meta fields written by the patched capture pipeline:
  capture_block in {"A_placement","B_eyetohand"}, cube_gripped (bool), grasp_id (int).
Frames with no tag are treated as A_placement (backward compatible).

Run:
  python Step3_compare_a_vs_b_base.py --root_folder ./data/session --intrinsics_dir ./intrinsics
  python Step3_compare_a_vs_b_base.py --selftest        # prove the eye-to-hand math
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import Step3_compare_calibrartion as S
from aruco_cube import ArucoCubeTarget
from config import get_default_cube_config
from calibration_runtime_utils import (
    get_capture_set_index, load_intrinsics_with_depth_scale,
    resolve_cube_config_for_run,
)
from cube_config_utils import cube_configs_equivalent, load_cube_config_from_meta
from robot_comm import euler_deg_to_matrix

inv = S.inv_T


def mk(Rm, t):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = np.asarray(t).ravel(); return T


def ang(M):
    return float(np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1))))


def trans_mm(M):
    return float(np.linalg.norm(M[:3, 3]) * 1000.0)


# ───────────────────────── eye-to-hand solver ─────────────────────────
def solve_eye_to_hand(Tbg_list, TCc_list):
    """Fixed camera Ci observes a cube rigidly on the gripper.

        T_Ci_cube(e) = T_Ci_base @ T_base_gripper(e) @ T_gripper_cube

    Mapped to cv2.calibrateRobotWorldHandEye's  A(e)·Y = X·B(e)  by:
        A = world2cam = T_Ci_cube(e),  B = base2gripper-input = T_base_gripper(e)
        Y = base2world = T_cube_gripper      -> T_gripper_cube = inv(Y)
        X = gripper2cam = T_Ci_base          -> T_base_Ci      = inv(X)
    (Derivation validated against the eye-in-hand convention + selftest below.)

    Returns (T_base_Ci, T_gripper_cube, median_trans_mm, median_rot_deg).
    """
    Rw = [T[:3, :3] for T in TCc_list]; tw = [T[:3, 3] for T in TCc_list]
    Rb = [T[:3, :3] for T in Tbg_list]; tb = [T[:3, 3] for T in Tbg_list]
    best = None
    for method in (cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH, cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI):
        try:
            Rbw, tbw, Rgc, tgc = cv2.calibrateRobotWorldHandEye(Rw, tw, Rb, tb, method=method)
        except cv2.error:
            continue
        Y = mk(Rbw, tbw)       # T_cube_gripper
        X = mk(Rgc, tgc)       # T_Ci_base
        rt, rr = [], []
        for i in range(len(TCc_list)):
            err = inv(TCc_list[i] @ Y) @ (X @ Tbg_list[i])   # both = T_Ci_gripper(e)
            rt.append(trans_mm(err)); rr.append(ang(err))
        score = np.median(rt) + 10 * np.median(rr)
        if best is None or score < best[0]:
            best = (score, inv(X), inv(Y), float(np.median(rt)), float(np.median(rr)))
    if best is None:
        return None, None, float("inf"), float("inf")
    return best[1], best[2], best[3], best[4]


def selftest():
    rng = np.random.default_rng(0)

    def rand_T(tscale=0.3):
        T = np.eye(4); T[:3, :3] = R.random(random_state=rng).as_matrix()
        T[:3, 3] = rng.uniform(-tscale, tscale, 3); return T

    T_base_Ci = rand_T(0.8); T_gripper_cube = rand_T(0.1)
    T_Ci_base = inv(T_base_Ci)
    Tbg, TCc = [], []
    for _ in range(15):
        B = rand_T(0.4)                       # T_base_gripper
        Tbg.append(B)
        TCc.append(T_Ci_base @ B @ T_gripper_cube)   # T_Ci_cube
    est_base_Ci, est_gc, mt, mr = solve_eye_to_hand(Tbg, TCc)
    d = inv(T_base_Ci) @ est_base_Ci
    print("[selftest] eye-to-hand recovery on synthetic exact data:")
    print(f"  residual: trans={mt:.4f}mm rot={mr:.4f}deg")
    print(f"  T_base_Ci error: Δt={trans_mm(d):.4f}mm Δrot={ang(d):.4f}deg")
    ok = mt < 1e-3 and ang(d) < 1e-2
    print("  => PASS (convention correct)" if ok else "  => FAIL")
    return ok


# ───────────────────────── data loading ─────────────────────────
def build_T_base_gripper(cap):
    m = cap.get("robot_pose_matrix_4x4")
    if m is not None:
        return np.asarray(m, dtype=np.float64)
    p = cap.get("robot_pose_6dof") or cap.get("capture_gripper_pose_6dof")
    return euler_deg_to_matrix(*[float(x) for x in p]) if p else None


def event_block(cap):
    b = cap.get("capture_block")
    if b in ("A_placement", "B_eyetohand"):
        return b
    g = cap.get("cube_gripped")
    if g is True:
        return "B_eyetohand"
    return "A_placement"   # default / legacy


def main():
    ap = argparse.ArgumentParser(description="Compare set_cube_center vs eye-to-hand base registration")
    ap.add_argument("--root_folder")
    ap.add_argument("--intrinsics_dir")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gripper_cam_idx", type=int, default=None)
    ap.add_argument("--ref_fixed_cam_idx", type=int, default=None)
    ap.add_argument("--max_err_fixed", type=float, default=3.0)
    ap.add_argument("--prior_max_trans_error_mm", type=float, default=100.0)
    ap.add_argument("--min_sweep_poses", type=int, default=6,
                    help="Min B_eyetohand poses per grasp/camera to attempt eye-to-hand.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        raise SystemExit(0 if ok else 1)
    if not (args.root_folder and args.intrinsics_dir):
        ap.error("--root_folder and --intrinsics_dir are required (or use --selftest)")

    root = args.root_folder
    out_dir = S.ensure_dir(args.out_dir or os.path.join(root, "calib_ablation"))
    meta = json.load(open(os.path.join(root, "meta.json")))

    cfg, _ = resolve_cube_config_for_run(root, out_dir, None, get_default_cube_config())
    meta_cfg, _ = load_cube_config_from_meta(root, default_cfg=cfg)
    reuse = cube_configs_equivalent(meta_cfg, cfg)
    cube = ArucoCubeTarget(cfg)

    all_ids = sorted({int(k) for cap in meta.get("captures", [])
                      for k, v in cap.get("cams", {}).items() if v.get("saved")})
    grip = args.gripper_cam_idx if args.gripper_cam_idx is not None else meta.get("gripper_cam_idx")
    fixed = [c for c in all_ids if c != int(grip)]
    ref = args.ref_fixed_cam_idx if args.ref_fixed_cam_idx is not None else fixed[0]
    K, D = {}, {}
    for ci in all_ids:
        K[ci], D[ci], _ = load_intrinsics_with_depth_scale(args.intrinsics_dir, ci)

    # split events by block
    A_events, B_events, e2s, e2cap = set(), set(), {}, {}
    Tbg = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        e2cap[eid] = cap
        s = get_capture_set_index(cap); e2s[eid] = int(s) if s is not None else None
        Tbg[eid] = build_T_base_gripper(cap)
        (B_events if event_block(cap) == "B_eyetohand" else A_events).add(eid)
    print(f"[INFO] cams={all_ids} fixed={fixed} ref=cam{ref} gripper=cam{grip}")
    print(f"[INFO] A_placement events={len(A_events)} | B_eyetohand events={len(B_events)}")

    # all fixed-cam cube observations (one pass)
    pose_obs = S.load_pose_observations(
        root=root, meta=meta, cube=cube, K_map=K, D_map=D, all_cam_ids=all_ids,
        gripper_cam_idx=int(grip), reuse_stored_cube_candidates=reuse,
        max_err_fixed=float(args.max_err_fixed), max_err_gripper=5.0,
        min_aspect_fixed=0.0, min_aspect_gripper=0.35, gripper_min_markers=1,
    )
    fixed_obs = [o for o in pose_obs if o.cam in fixed]

    result = {"n_A": len(A_events), "n_B": len(B_events)}

    # ── (a) set_cube_center anchor on A_placement frames ──
    T_base_Ci_a = None
    a_rms = None
    A_obs = [o for o in fixed_obs if o.event in A_events]
    set_priors = {}
    for cap in meta.get("captures", []):
        if int(cap.get("event_id", -1)) in A_events:
            s = get_capture_set_index(cap)
            raw = cap.get("set_cube_center_6dof")
            if s is not None and raw is not None and int(s) not in set_priors:
                set_priors[int(s)] = S.pose6_to_T_base_gripper([float(x) for x in raw])
    if A_obs and len(set_priors) >= 3:
        set_pose6 = {s: [float(x) for x in cap["set_cube_center_6dof"]]
                     for cap in meta["captures"]
                     for s in [get_capture_set_index(cap)]
                     if s is not None and cap.get("set_cube_center_6dof") is not None}
        T_base_ref, T_base_Ci_a, _, diag_a, _, stats_a = S.initialize_base_translation_anchored(
            A_obs, fixed, ref, set_priors, set_pose6, e2s,
            max_trans_error_mm=float(args.prior_max_trans_error_mm),
            max_rot_error_deg=45.0, disable_if_inconsistent=True)
        a_rms = diag_a.get("anchor_rms_mm")
        print(f"[a] set_cube_center anchor: {len(set_priors)} sets, Kabsch rms={a_rms:.1f}mm")
        result["a"] = {"anchor_rms_mm": a_rms, "n_sets": len(set_priors), "stats": stats_a}
    else:
        print(f"[a] SKIP: need >=3 placement sets with set_cube_center (have {len(set_priors)})")
        result["a"] = {"skipped": "insufficient A_placement / set_cube_center data"}

    # ── (b) eye-to-hand on B_eyetohand frames ──
    T_base_Ci_b = None
    b_res = {}
    if B_events:
        by_cam_grasp = defaultdict(lambda: defaultdict(list))   # cam -> grasp -> [events]
        for o in fixed_obs:
            if o.event in B_events:
                g = int(e2cap[o.event].get("grasp_id", 0))
                by_cam_grasp[o.cam][g].append(o)
        T_base_Ci_b = {}
        for ci in fixed:
            per_grasp = []
            for g, obs in sorted(by_cam_grasp.get(ci, {}).items()):
                obs = [o for o in obs if Tbg.get(o.event) is not None]
                if len(obs) < args.min_sweep_poses:
                    continue
                Tbg_l = [Tbg[o.event] for o in obs]
                TCc_l = [o.T_C_O for o in obs]          # cube in cam Ci
                T_bc, _, mt, mr = solve_eye_to_hand(Tbg_l, TCc_l)
                if T_bc is not None:
                    per_grasp.append((T_bc, mt, mr, len(obs), g))
            if per_grasp:
                T_avg = S.weighted_se3_average([p[0] for p in per_grasp])
                T_base_Ci_b[ci] = T_avg
                mt = float(np.median([p[1] for p in per_grasp]))
                mr = float(np.median([p[2] for p in per_grasp]))
                b_res[ci] = {"residual_trans_mm": mt, "residual_rot_deg": mr,
                             "n_grasps": len(per_grasp), "poses": [p[3] for p in per_grasp]}
                flag = "OK" if (mt <= 5 and mr <= 1.5) else "UNRELIABLE"
                print(f"[b] cam{ci}: eye-to-hand residual {mt:.1f}mm/{mr:.2f}deg "
                      f"({len(per_grasp)} grasp(s)) -> {flag}")
            else:
                print(f"[b] cam{ci}: no grasp had >= {args.min_sweep_poses} sweep poses")
        result["b"] = b_res if b_res else {"skipped": "no usable eye-to-hand sequences"}
    else:
        print("[b] SKIP: no B_eyetohand frames. Capture Block B (cube gripped, robot sweep) "
              "with `block b` in the server; see _diag_handeye.py for the motion-richness check.")
        result["b"] = {"skipped": "no B_eyetohand frames in this session"}

    # ── comparison ──
    print("\n" + "=" * 78)
    print("BASE REGISTRATION: (a) set_cube_center  vs  (b) eye-to-hand")
    print("=" * 78)
    hdr = f"{'cam':>4s} | {'(a) rms':>8s} | {'(b) residual':>14s} | {'Δt (a vs b)':>11s} | {'Δrot':>7s}"
    print(hdr); print("-" * len(hdr))
    cmp_rows = {}
    for ci in fixed:
        a_str = f"{a_rms:.1f}mm" if a_rms is not None else "NA"
        if ci in b_res:
            b_str = f"{b_res[ci]['residual_trans_mm']:.1f}/{b_res[ci]['residual_rot_deg']:.2f}"
        else:
            b_str = "NA"
        if T_base_Ci_a and T_base_Ci_b and ci in T_base_Ci_a and ci in T_base_Ci_b:
            d = inv(T_base_Ci_a[ci]) @ T_base_Ci_b[ci]
            dt, dr = trans_mm(d), ang(d)
            cmp_rows[ci] = {"delta_trans_mm": dt, "delta_rot_deg": dr}
            print(f"{ci:>4d} | {a_str:>8s} | {b_str:>14s} | {dt:8.1f}mm | {dr:5.2f}°")
        else:
            print(f"{ci:>4d} | {a_str:>8s} | {b_str:>14s} | {'NA':>11s} | {'NA':>7s}")
    result["comparison"] = cmp_rows
    if not cmp_rows:
        print("\n(no (a)-vs-(b) comparison yet — need BOTH A_placement and valid B_eyetohand data)")

    with open(os.path.join(out_dir, "compare_a_vs_b_base.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[DONE] {os.path.join(out_dir, 'compare_a_vs_b_base.json')}")


if __name__ == "__main__":
    main()
