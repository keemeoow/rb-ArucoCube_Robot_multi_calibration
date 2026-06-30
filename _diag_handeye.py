#!/usr/bin/env python3
"""Hand-eye vs no-hand-eye comparison for camera<->robot-base registration.

without hand-eye: base frame from set_cube_center prior (Kabsch on positions).
with hand-eye   : base frame from robot FK + gripper-cam, INDEPENDENT of the prior.

Model (robot-world hand-eye):  A(e) = Y @ B(e) @ X
  A(e) = T_cam2_cam0(e)  (gripper cam pose seen via fixed-cam vision)
  B(e) = T_gripper_base(e) = inv(robot FK)
  Y = T_cam0_base (base registration we want), X = T_cam2_gripper (mount)
Solved with cv2.calibrateRobotWorldHandEye, then validated by per-event residual.
"""
import json, pickle
import numpy as np
import cv2
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

import Step3_compare_calibrartion as S
from calibration_runtime_utils import get_capture_set_index
from robot_comm import euler_deg_to_matrix

meta = json.load(open("data/session/meta.json"))
pose_obs = pickle.load(open("_diag_pose_obs.pkl", "rb"))
grip = int(meta["gripper_cam_idx"])
all_ids = sorted({int(k) for c in meta["captures"] for k, v in c["cams"].items() if v.get("saved")})
fixed = [c for c in all_ids if c != grip]
ref = fixed[0]
inv = S.inv_T

def ang(M):
    c = np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))

# robot FK per event + set index
T_base_grip = {}
e2s = {}
for cap in meta["captures"]:
    eid = int(cap.get("event_id", -1))
    if eid < 0:
        continue
    s = get_capture_set_index(cap)
    e2s[eid] = int(s) if s is not None else None
    m = cap.get("robot_pose_matrix_4x4")
    if m is not None:
        T_base_grip[eid] = np.asarray(m, dtype=np.float64)
    else:
        T_base_grip[eid] = euler_deg_to_matrix(*cap["robot_pose_6dof"])

# sanity: matrix vs euler
e0 = sorted(T_base_grip)[0]
chk = euler_deg_to_matrix(*meta["captures"][0]["robot_pose_6dof"])
print(f"[chk] robot_pose_matrix_4x4 vs euler(pose6): max|Δ|={np.max(np.abs(T_base_grip[e0]-chk)):.2e} "
      f"(translation in m? t={T_base_grip[e0][:3,3]})")

# fixed-cam vision: cube in cam0 per event
fixed_pose = [o for o in pose_obs if o.cam in fixed]
T_cam0_C, _ = S.build_ref_relative_from_pairwise(fixed_pose, fixed, ref, robust=True)
T_cam0_O = S.initialize_ref_object_poses(fixed_pose, T_cam0_C, fixed, ref)

# gripper cam cube pose per event
T_cam2_O = {o.event: o.T_C_O for o in pose_obs if o.cam == grip}

events = sorted(set(T_cam0_O) & set(T_cam2_O) & set(T_base_grip))
print(f"[info] events usable for hand-eye: {len(events)}")

# Build A(e)=T_cam2_cam0(e), B(e)=T_gripper_base(e)=inv(FK)
R_w2c, t_w2c, R_b2g, t_b2g = [], [], [], []
for e in events:
    T_cam2_cam0 = T_cam2_O[e] @ inv(T_cam0_O[e])   # world(cam0) -> cam(cam2)
    T_grip_base = inv(T_base_grip[e])              # base -> gripper
    R_w2c.append(T_cam2_cam0[:3, :3]); t_w2c.append(T_cam2_cam0[:3, 3])
    R_b2g.append(T_grip_base[:3, :3]); t_b2g.append(T_grip_base[:3, 3])

def solve_and_validate(method, name):
    R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        R_w2c, t_w2c, R_b2g, t_b2g, method=method)
    Y = np.eye(4); Y[:3, :3] = R_b2w; Y[:3, 3] = t_b2w.ravel()       # base2world = T_cam0_base
    Xg2c = np.eye(4); Xg2c[:3, :3] = R_g2c; Xg2c[:3, 3] = t_g2c.ravel()  # gripper2cam = T_cam2_gripper
    # validate: world2cam(e) @ base2world  ==  gripper2cam @ base2gripper(e)   (both base->cam)
    rt, rr = [], []
    for i, e in enumerate(events):
        Twc = np.eye(4); Twc[:3, :3] = R_w2c[i]; Twc[:3, 3] = t_w2c[i]
        B = inv(T_base_grip[e])                  # base2gripper
        lhs = Twc @ Y
        rhs = Xg2c @ B
        err = inv(lhs) @ rhs
        rt.append(np.linalg.norm(err[:3, 3]) * 1000); rr.append(ang(err))
    print(f"[{name}] residual: trans med={np.median(rt):.1f}mm | rot med={np.median(rr):.2f}deg")
    return Y, Xg2c, np.median(rt), np.median(rr)

best = None
for m, nm in [(cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH, "Shah"), (cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI, "Li")]:
    Y, Xg2c, mt, mr = solve_and_validate(m, nm)
    if best is None or (mt + mr * 10) < best[2]:
        best = (Y, Xg2c, mt + mr * 10, mt, mr)
Y, Xg2c = best[0], best[1]
print(f"[validate] best robot-world hand-eye residual: trans med={best[3]:.1f}mm | rot med={best[4]:.2f}deg")
T_base_cam0_he = inv(Y)
T_gripper_cam2 = inv(Xg2c)
print(f"[hand-eye] T_gripper_cam2 translation (mm)={T_gripper_cam2[:3,3]*1000}  rot(deg)={R.from_matrix(T_gripper_cam2[:3,:3]).as_euler('ZYX',degrees=True)}")

# ── without hand-eye: prior (Kabsch) base anchor ──
set_priors = S.load_nominal_set_cube_transforms(meta)
by_set = defaultdict(list)
for e, T in T_cam0_O.items():
    s = e2s.get(e)
    if s in set_priors:
        by_set[s].append(T)
set_vis = {s: S.weighted_se3_average(v) for s, v in by_set.items()}
sets = sorted(set_vis)
src = np.array([set_vis[s][:3, 3] for s in sets])     # cam0
dst = np.array([set_priors[s][:3, 3] for s in sets])  # base
T_base_ref_prior, keep = S.robust_kabsch_rigid(src, dst, max_resid_mm=100.0)
T_base_cam0_prior = T_base_ref_prior

# ── compare the two base registrations ──
print()
print("================ base registration: WITHOUT vs WITH hand-eye ================")
d = inv(T_base_cam0_prior) @ T_base_cam0_he
print(f"T_base_cam0 difference (prior-anchor vs hand-eye-anchor): "
      f"Δtrans={np.linalg.norm(d[:3,3])*1000:.1f}mm  Δrot={ang(d):.2f}deg")

# per fixed camera T_base_Ci difference
print("per fixed-camera T_base_Ci difference:")
for ci in fixed:
    Ta = T_base_cam0_prior @ T_cam0_C[ci]
    Tb = T_base_cam0_he @ T_cam0_C[ci]
    dd = inv(Ta) @ Tb
    print(f"  cam{ci}: Δtrans={np.linalg.norm(dd[:3,3])*1000:6.1f}mm  Δrot={ang(dd):5.2f}deg")

# cube-center positions: prior vs hand-eye-derived (validate the ~8mm prior error)
print("\ncube-center position: set_cube_center(prior) vs hand-eye-derived (base, mm):")
he_t, pr_t = [], []
for s in sets:
    # hand-eye cube pose in base from gripper cam, per event mean
    Ts = []
    for e in events:
        if e2s.get(e) == s:
            T_base_cam2 = T_base_grip[e] @ T_gripper_cam2
            Ts.append(T_base_cam2 @ T_cam2_O[e])
    if not Ts:
        continue
    he = S.weighted_se3_average(Ts)
    pr = set_priors[s]
    dpos = np.linalg.norm(he[:3, 3] - pr[:3, 3]) * 1000
    he_t.append(dpos)
    print(f"  set{s:2d}: |hand-eye - prior| = {dpos:5.1f}mm")
print(f"  => median |hand-eye cube-center - set_cube_center prior| = {np.median(he_t):.1f}mm")
