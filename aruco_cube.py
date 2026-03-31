# aruco_cube.py
"""
ArUco cube target: geometry model, detection, solvePnP.
Reusable across all calibration steps.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from config import CubeConfig


# ─────────────────────────── utils ───────────────────────────

def rodrigues_to_Rt(rvec, tvec) -> np.ndarray:
    """OpenCV rvec,tvec -> 4x4 T_C_O (Object->Camera)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid-body transform."""
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula: axis-angle -> rotation matrix."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ─────────────────────── Cube Geometry ───────────────────────

class ArucoCubeModel:
    """3D geometry of a cube with ArUco markers on 5 faces."""

    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        d = cfg.cube_side_m / 2.0
        s = cfg.marker_size_m / 2.0

        # face definitions: (center, u-axis, v-axis, normal) in object frame
        self.face_defs = {
            "+Z": (np.array([0, 0, d]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
            "-Z": (np.array([0, 0, -d]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1])),
            "+X": (np.array([d, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([1, 0, 0])),
            "-X": (np.array([-d, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([-1, 0, 0])),
            "+Y": (np.array([0, d, 0]), np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0])),
        }

        self.local_corners = np.array([
            [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]
        ], dtype=np.float64)

    def marker_corners_in_rig(self, marker_id: int) -> np.ndarray:
        """4 corners of marker_id in the cube (object/rig) frame. Shape (4,3)."""
        face = self.cfg.id_to_face[int(marker_id)]
        c, u, v, n = self.face_defs[face]

        roll = np.deg2rad(float(self.cfg.face_roll_deg.get(int(marker_id), 0.0)))
        Rr = rot_axis_angle(n, roll)
        u2 = (Rr @ u.reshape(3, 1)).reshape(3)
        v2 = (Rr @ v.reshape(3, 1)).reshape(3)

        pts = []
        for p in self.local_corners:
            pts.append(c + u2 * p[0] + v2 * p[1])
        return np.asarray(pts, dtype=np.float64)


# ──────────────────── Detection + PnP ────────────────────────

class ArucoCubeTarget:
    """Full pipeline: detect markers -> build 2D-3D correspondences -> solvePnP."""

    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        self.model = ArucoCubeModel(cfg)

        d = getattr(cv2.aruco, cfg.dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(d)

        # Compatibility: OpenCV 4.7+ has ArucoDetector, older uses detectMarkers
        try:
            self.params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)
            self._use_new_api = True
        except AttributeError:
            self.params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False

    def detect(self, bgr: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Return (corners_list, ids_flat_or_None)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        if ids is None:
            return [], None
        return corners, ids.flatten().astype(int)

    def build_correspondences(self, corners_list, ids, min_markers: int = 1,
                              only_ids: Optional[List[int]] = None,
                              min_aspect: float = 0.3):
        """Build 2D-3D correspondences from detected markers.
        min_aspect: reject markers with aspect ratio below this (0=no filter).
        Returns (obj_pts, img_pts, used_ids) or (None, None, [])."""
        obj_pts, img_pts, used = [], [], []
        only_set = set(only_ids) if only_ids is not None else None

        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in self.cfg.id_to_face:
                continue
            if only_set is not None and mid not in only_set:
                continue

            obj = self.model.marker_corners_in_rig(mid)
            img = c.reshape(4, 2)

            # Corner reorder per marker (from config)
            reorder = getattr(self.cfg, 'corner_reorder', {}).get(mid)
            if reorder is not None:
                img = img[reorder]

            # Skip markers seen at extreme oblique angles (nearly edge-on)
            if min_aspect > 0:
                edge_w = np.linalg.norm(img[1] - img[0])
                edge_h = np.linalg.norm(img[3] - img[0])
                aspect = min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6)
                if aspect < min_aspect:
                    continue

            obj_pts.append(obj)
            img_pts.append(img)
            used.append(mid)

        if len(used) < min_markers:
            return None, None, used

        obj_pts = np.concatenate(obj_pts).reshape(-1, 1, 3).astype(np.float64)
        img_pts = np.concatenate(img_pts).reshape(-1, 1, 2).astype(np.float64)
        return obj_pts, img_pts, used

    def solve_pnp_cube(self, bgr, K, D,
                       use_ransac: bool = True,
                       min_markers: int = 1,
                       reproj_thr_mean_px: float = 10.0,
                       only_ids: Optional[List[int]] = None,
                       return_reproj: bool = False,
                       min_aspect: float = 0.3):
        """
        Full detect + PnP solve.
        min_aspect: reject oblique markers (0=no filter, 0.3=default).
        Returns:
          (ok, rvec, tvec, used_ids)  if return_reproj=False
          (ok, rvec, tvec, used_ids, reproj_dict)  if return_reproj=True
        """
        corners_list, ids = self.detect(bgr)
        if ids is None:
            return (False, None, None, [], None) if return_reproj else (False, None, None, [])

        obj_pts, img_pts, used = self.build_correspondences(
            corners_list, ids, min_markers, only_ids, min_aspect=min_aspect)
        if obj_pts is None:
            return (False, None, None, used, None) if return_reproj else (False, None, None, used)

        n = int(obj_pts.shape[0])
        flags = cv2.SOLVEPNP_ITERATIVE if n >= 8 else cv2.SOLVEPNP_IPPE

        if use_ransac and n >= 8:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj_pts, img_pts, K, D, flags=flags,
                reprojectionError=5.0, iterationsCount=200, confidence=0.999)
        else:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flags)

        if not ok:
            return (False, None, None, used, None) if return_reproj else (False, None, None, used)

        proj2, _ = cv2.projectPoints(obj_pts.reshape(-1, 3), rvec, tvec, K, D)
        proj2 = proj2.reshape(-1, 2)
        err = np.linalg.norm(proj2 - img_pts.reshape(-1, 2), axis=1)

        reproj = {
            "obj_pts": obj_pts, "img_pts": img_pts, "proj2": proj2, "err": err,
            "err_mean": float(np.mean(err)), "err_median": float(np.median(err)),
            "err_p90": float(np.percentile(err, 90)), "n_points": int(err.size),
            "rvec": rvec, "tvec": tvec,
        }
        ok_final = reproj["err_mean"] <= reproj_thr_mean_px

        if return_reproj:
            return ok_final, rvec, tvec, used, reproj
        return ok_final, rvec, tvec, used
