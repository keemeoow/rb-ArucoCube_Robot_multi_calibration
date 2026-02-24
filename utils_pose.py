# utils_pose.py
"""
SE(3) utilities: quaternion averaging, robust mean, distance metric.
"""

import numpy as np


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w,x,y,z), numerically stable."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, x, y, z = (R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, x, y, z = (R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, x, y, z = (R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> rotation matrix."""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ], dtype=np.float64)


def _rot_angle(R: np.ndarray) -> float:
    """Geodesic rotation angle (rad)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


def _average_quats(quats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Markley method: eigen of weighted outer products."""
    A = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quats, weights):
        q = q.reshape(4, 1)
        A += w * (q @ q.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[0] < 0:
        q_avg = -q_avg
    return q_avg / (np.linalg.norm(q_avg) + 1e-12)


def se3_distance(Ta: np.ndarray, Tb: np.ndarray, w_rot=1.0, w_trans=1.0) -> float:
    """Simple SE(3) distance: geodesic angle + translation norm."""
    dR = Ta[:3,:3] @ Tb[:3,:3].T
    return float(w_rot * _rot_angle(dR) + w_trans * np.linalg.norm(Ta[:3,3] - Tb[:3,3]))


def robust_se3_average(T_list, max_iters=5, k_mad=2.5, return_stats=False):
    """
    Robust SE(3) mean via iterative MAD-based outlier rejection.
    Returns T_mean, or (T_mean, stats_dict) if return_stats=True.
    """
    if len(T_list) == 0:
        raise ValueError("T_list is empty")
    Ts = [np.asarray(T, dtype=np.float64) for T in T_list]

    def compute_mean(Ts_in):
        quats = np.array([_R_to_quat(T[:3,:3]) for T in Ts_in])
        trans = np.array([T[:3,3] for T in Ts_in])
        # align quaternion signs
        for i in range(len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]
        q_mean = _average_quats(quats, np.ones(len(quats)))
        Tm = np.eye(4, dtype=np.float64)
        Tm[:3,:3] = _quat_to_R(q_mean)
        Tm[:3,3] = np.mean(trans, axis=0)
        return Tm

    T_mean = compute_mean(Ts)
    inlier_mask = np.ones(len(Ts), dtype=bool)

    for _ in range(max_iters):
        res = np.array([se3_distance(T, T_mean) for T in Ts])
        med = np.median(res)
        mad = np.median(np.abs(res - med)) + 1e-12
        thr = med + k_mad * 1.4826 * mad
        new_mask = res <= thr
        if new_mask.sum() < max(3, int(0.3 * len(Ts))):
            break
        inlier_mask = new_mask
        T_new = compute_mean([T for T, m in zip(Ts, inlier_mask) if m])
        if se3_distance(T_new, T_mean) < 1e-6:
            T_mean = T_new
            break
        T_mean = T_new

    if not return_stats:
        return T_mean

    rot_devs = [_rot_angle(T[:3,:3] @ T_mean[:3,:3].T) * 180/np.pi for T in Ts]
    trans_devs = [np.linalg.norm(T[:3,3] - T_mean[:3,3]) * 1000 for T in Ts]
    stats = {
        "num_frames": len(Ts), "num_inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "rotation_std_deg": float(np.std(rot_devs)),
        "translation_std_mm": float(np.std(trans_devs)),
    }
    return T_mean, stats
