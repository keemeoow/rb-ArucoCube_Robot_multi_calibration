#!/usr/bin/env python3
"""Root-cause diagnostic for the with-prior calibration breakdown.

Tests whether set_cube_center_6dof priors are consistent with the vision
geometry, and whether any frame-convention reinterpretation fixes them.
Caches pose observations so it is fast to re-run.
"""
import json, os, pickle
import numpy as np

import Step3_compare_calibrartion as S
from aruco_cube import ArucoCubeTarget
from config import get_default_cube_config
from calibration_runtime_utils import (
    load_intrinsics_with_depth_scale, resolve_cube_config_for_run, get_capture_set_index,
)
from cube_config_utils import cube_configs_equivalent, load_cube_config_from_meta

ROOT = "./data/session"
INTR = "./intrinsics"
CACHE = "./_diag_pose_obs.pkl"

meta = json.load(open(os.path.join(ROOT, "meta.json")))
cfg, _ = resolve_cube_config_for_run(ROOT, "./data/session/calib_ablation", None, get_default_cube_config())
meta_cfg, _ = load_cube_config_from_meta(ROOT, default_cfg=cfg)
reuse_stored = cube_configs_equivalent(meta_cfg, cfg)
cube = ArucoCubeTarget(cfg)

all_cam_ids = sorted({int(k) for cap in meta.get("captures", []) for k, v in cap.get("cams", {}).items() if v.get("saved")})
gripper = int(meta.get("gripper_cam_idx"))
fixed = [c for c in all_cam_ids if c != gripper]
ref = fixed[0]
K, D = {}, {}
for ci in all_cam_ids:
    K[ci], D[ci], _ = load_intrinsics_with_depth_scale(INTR, ci)

event_to_set = {}
for cap in meta.get("captures", []):
    eid = int(cap.get("event_id", -1))
    if eid >= 0:
        s = get_capture_set_index(cap)
        event_to_set[eid] = int(s) if s is not None else None

if os.path.exists(CACHE):
    pose_obs = pickle.load(open(CACHE, "rb"))
    print(f"[cache] loaded {len(pose_obs)} pose obs")
else:
    pose_obs = S.load_pose_observations(
        root=ROOT, meta=meta, cube=cube, K_map=K, D_map=D, all_cam_ids=all_cam_ids,
        gripper_cam_idx=gripper, reuse_stored_cube_candidates=reuse_stored,
        max_err_fixed=3.0, max_err_gripper=5.0, min_aspect_fixed=0.0, min_aspect_gripper=0.35,
        gripper_min_markers=1,
    )
    pickle.dump(pose_obs, open(CACHE, "wb"))
    print(f"[built] cached {len(pose_obs)} pose obs")

fixed_pose = [o for o in pose_obs if o.cam in fixed]

# Vision-only calibration (no prior), cam0 reference frame.
T_ref_C, _ = S.build_ref_relative_from_pairwise(fixed_pose, fixed, ref, robust=True)
T_ref_O = S.initialize_ref_object_poses(fixed_pose, T_ref_C, fixed, ref)  # object-in-cam0 per event

set_priors = S.load_nominal_set_cube_transforms(meta)  # P[s] from pose6_to_T (interpreted T_base_O)
print(f"[info] cams={all_cam_ids} fixed={fixed} ref=cam{ref} | events with vision pose={len(T_ref_O)} | prior sets={sorted(set_priors)}")


def rot_deg(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def spread(T_list):
    """Translation std (mm) and rotation std (deg) of a list of SE3 about their mean."""
    Tm = S.weighted_se3_average(T_list)
    tr = [np.linalg.norm(T[:3, 3] - Tm[:3, 3]) * 1000 for T in T_list]
    ro = [rot_deg(T[:3, :3] @ Tm[:3, :3].T) for T in T_list]
    return float(np.std(tr)), float(np.std(ro)), float(np.mean(tr)), float(np.mean(ro))


inv = S.inv_T

# Candidate interpretations of the prior P relative to vision object pose T_ref_O.
# If correct, T_base_ref(e) = f(P) @ inv(T_ref_O_e) is CONSTANT across events.
candidates = {
    "P as T_base_O          : P @ inv(T_ref_O)": lambda P, To: P @ inv(To),
    "inv(P) as T_base_O     : invP @ inv(T_ref_O)": lambda P, To: inv(P) @ inv(To),
    "P as T_base_O, obj inv : P @ To": lambda P, To: P @ To,
    "inv(P), obj inv        : invP @ To": lambda P, To: inv(P) @ To,
}

print("\n=== Consistency of implied T_base_cam0 across events (should be ~0 std if prior valid) ===")
for name, f in candidates.items():
    Ts = []
    for eid, To in T_ref_O.items():
        s = event_to_set.get(eid)
        if s is None or s not in set_priors:
            continue
        Ts.append(f(set_priors[s], To))
    if len(Ts) < 3:
        print(f"  {name:48s} -> only {len(Ts)} events")
        continue
    tstd, rstd, tm, rm = spread(Ts)
    print(f"  {name:48s} -> n={len(Ts):3d}  trans_std={tstd:8.1f}mm  rot_std={rstd:7.2f}deg")

# Per-set: are events WITHIN one set self-consistent (vision relative motion vs static prior)?
print("\n=== Within-set vision object-pose spread (cube is physically static within a set) ===")
from collections import defaultdict
by_set = defaultdict(list)
for eid, To in T_ref_O.items():
    s = event_to_set.get(eid)
    if s is not None:
        by_set[s].append(To)
for s in sorted(by_set):
    if len(by_set[s]) < 2:
        continue
    tstd, rstd, tm, rm = spread(by_set[s])
    has = "prior" if s in set_priors else "no-prior"
    print(f"  set {s:2d} ({has:8s}) n={len(by_set[s]):2d}  vision obj trans_std={tstd:7.1f}mm  rot_std={rstd:6.2f}deg")

# Relative test: prior relative pose between consecutive prior-sets vs vision relative pose.
print("\n=== Relative pose between prior sets: PRIOR vs VISION ===")
# vision per-set mean object pose
vis_set = {}
for s, lst in by_set.items():
    if s in set_priors and len(lst) >= 1:
        vis_set[s] = S.weighted_se3_average(lst)
sets = sorted(vis_set)
print(f"  prior sets with vision: {sets}")
for i in range(len(sets) - 1):
    a, b = sets[i], sets[i + 1]
    rel_prior = inv(set_priors[a]) @ set_priors[b]
    rel_vis = inv(vis_set[a]) @ vis_set[b]
    dt = np.linalg.norm(rel_prior[:3, 3] - rel_vis[:3, 3]) * 1000
    dr = rot_deg(rel_prior[:3, :3] @ rel_vis[:3, :3].T)
    pt = np.linalg.norm(rel_prior[:3, 3]) * 1000
    vt = np.linalg.norm(rel_vis[:3, 3]) * 1000
    print(f"  set{a}->set{b}: |prior_rel_t|={pt:6.1f}mm |vis_rel_t|={vt:6.1f}mm  Δt={dt:6.1f}mm  Δrot={dr:6.2f}deg")

# Kabsch: rigidly align prior cube-CENTER positions (base) to vision centers (cam0).
# If residual is small, translation-only prior is a valid base-frame anchor.
print("\n=== Kabsch alignment of cube-center POSITIONS only (prior_base vs vision_cam0) ===")
sets_k = [s for s in sorted(vis_set) if s in set_priors]
Pbase = np.array([set_priors[s][:3, 3] for s in sets_k])      # base frame (m)
Vcam0 = np.array([vis_set[s][:3, 3] for s in sets_k])         # cam0 frame (m)
cb, cv = Pbase.mean(0), Vcam0.mean(0)
# Kabsch mapping source Vcam0 -> target Pbase:  R = Vt^T diag(1,1,d) U^T
H = (Vcam0 - cv).T @ (Pbase - cb)
U, _, Vt = np.linalg.svd(H)
d = np.sign(np.linalg.det(Vt.T @ U.T))
Rk = Vt.T @ np.diag([1, 1, d]) @ U.T          # rotates cam0 -> base
tk = cb - Rk @ cv
resid = [np.linalg.norm((Rk @ Vcam0[i] + tk) - Pbase[i]) * 1000 for i in range(len(sets_k))]
print(f"  PROPER (det=+1)   residual: mean={np.mean(resid):.1f}mm  max={np.max(resid):.1f}mm  rms={np.sqrt(np.mean(np.square(resid))):.1f}mm")
# pairwise distance congruence check (frame-independent)
import itertools
dmax = 0.0
for i, j in itertools.combinations(range(len(sets_k)), 2):
    dp = np.linalg.norm(Pbase[i] - Pbase[j]) * 1000
    dv = np.linalg.norm(Vcam0[i] - Vcam0[j]) * 1000
    dmax = max(dmax, abs(dp - dv))
print(f"  max |pairwise_dist_prior - pairwise_dist_vision| over all 78 pairs = {dmax:.1f}mm")
print("  prior centers (base, mm)        vision centers (cam0, mm)")
for i, s in enumerate(sets_k):
    pb = Pbase[i] * 1000; vc = Vcam0[i] * 1000
    print(f"    set{s:2d}: [{pb[0]:7.1f} {pb[1]:7.1f} {pb[2]:7.1f}]   [{vc[0]:7.1f} {vc[1]:7.1f} {vc[2]:7.1f}]")

print("\n=== Raw priors (T_base_O interpretation) ===")
for s in sorted(set_priors):
    P = set_priors[s]
    t = P[:3, 3] * 1000
    print(f"  set {s:2d}: t(mm)=[{t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f}]  R=\n{np.array2string(P[:3,:3], precision=3, suppress_small=True)}")
