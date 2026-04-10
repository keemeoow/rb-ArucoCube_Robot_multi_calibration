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


@dataclass
class MarkerObservation:
    marker_id: int
    face_name: str
    corners_2d: np.ndarray
    corners_2d_reordered: np.ndarray
    obj_corners_3d: np.ndarray
    aspect_ratio: float
    T_object_marker: np.ndarray


@dataclass
class CubeObservationSet:
    marker_observations: List[MarkerObservation]

    def image_bbox(self, pad_px: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        if not self.marker_observations:
            return None
        pts = np.concatenate([obs.corners_2d for obs in self.marker_observations], axis=0)
        x0 = int(np.floor(np.min(pts[:, 0]) - float(pad_px)))
        y0 = int(np.floor(np.min(pts[:, 1]) - float(pad_px)))
        x1 = int(np.ceil(np.max(pts[:, 0]) + float(pad_px)))
        y1 = int(np.ceil(np.max(pts[:, 1]) + float(pad_px)))
        return x0, y0, x1, y1


def depth_metrics_to_fields(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = metrics or {}

    def _maybe_float(key: str) -> Optional[float]:
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)

    return {
        "depth_valid": bool(metrics.get("valid", False)),
        "depth_num_samples": int(metrics.get("num_samples", 0)),
        "depth_num_markers": int(metrics.get("num_markers", 0)),
        "depth_support_marker_ids": [int(x) for x in metrics.get("support_marker_ids", [])],
        "depth_plane_mean_mm": _maybe_float("plane_mean_mm"),
        "depth_plane_median_mm": _maybe_float("plane_median_mm"),
        "depth_plane_max_mm": _maybe_float("plane_max_mm"),
        "depth_z_mean_mm": _maybe_float("z_mean_mm"),
        "depth_z_median_mm": _maybe_float("z_median_mm"),
        "depth_inlier_ratio": _maybe_float("inlier_ratio"),
    }


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
            "-Y": (np.array([0, -d, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, -1, 0])),
        }

        self.local_corners = np.array([
            [s, -s, 0], [-s, -s, 0], [-s, s, 0], [s, s, 0]
        ], dtype=np.float64)

    def has_marker(self, marker_id: int) -> bool:
        mid = int(marker_id)
        return mid in getattr(self.cfg, "marker_pose_4x4", {}) or mid in self.cfg.id_to_face

    def uses_explicit_marker_pose(self, marker_id: int) -> bool:
        return int(marker_id) in getattr(self.cfg, "marker_pose_4x4", {})

    def marker_face_name(self, marker_id: int) -> str:
        mid = int(marker_id)
        if mid in getattr(self.cfg, "marker_pose_4x4", {}):
            return str(self.cfg.id_to_face.get(mid, f"marker_{mid}"))
        return str(self.cfg.id_to_face[mid])

    def marker_pose_in_rig(self, marker_id: int) -> np.ndarray:
        mid = int(marker_id)
        if self.uses_explicit_marker_pose(mid):
            return np.asarray(self.cfg.marker_pose_4x4[mid], dtype=np.float64).reshape(4, 4)

        face = self.cfg.id_to_face[mid]
        c, u, v, n = self.face_defs[face]

        roll = np.deg2rad(float(self.cfg.face_roll_deg.get(mid, 0.0)))
        Rr = rot_axis_angle(n, roll)
        u2 = (Rr @ u.reshape(3, 1)).reshape(3)
        v2 = (Rr @ v.reshape(3, 1)).reshape(3)
        n2 = np.cross(u2, v2)
        n2 = n2 / (np.linalg.norm(n2) + 1e-12)

        T = np.eye(4, dtype=np.float64)
        T[:3, 0] = u2 / (np.linalg.norm(u2) + 1e-12)
        T[:3, 1] = v2 / (np.linalg.norm(v2) + 1e-12)
        T[:3, 2] = n2
        T[:3, 3] = c
        return T

    def marker_corners_in_rig(self, marker_id: int) -> np.ndarray:
        """4 corners of marker_id in the cube (object/rig) frame. Shape (4,3)."""
        T_OM = self.marker_pose_in_rig(marker_id)
        pts = []
        for p in self.local_corners:
            ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
            pts.append((T_OM @ ph)[:3])
        return np.asarray(pts, dtype=np.float64)

    def reorder_image_corners(self, marker_id: int, img_pts: np.ndarray) -> np.ndarray:
        mid = int(marker_id)
        out = np.asarray(img_pts, dtype=np.float64).reshape(4, 2)
        reorder = getattr(self.cfg, "corner_reorder", {}).get(mid)
        if reorder is not None:
            out = out[reorder]
        return out

    def marker_visibility_score(self, marker_id: int, T_C_O: np.ndarray) -> Tuple[bool, float]:
        T_O_M = self.marker_pose_in_rig(int(marker_id))
        center_obj = np.asarray(T_O_M[:3, 3], dtype=np.float64)
        normal_obj = np.asarray(T_O_M[:3, 2], dtype=np.float64)
        R = np.asarray(T_C_O[:3, :3], dtype=np.float64)
        t = np.asarray(T_C_O[:3, 3], dtype=np.float64)
        center_cam = R @ center_obj + t
        normal_cam = R @ normal_obj
        vis_score = float(np.dot(normal_cam, -center_cam))
        return bool(vis_score > 0.0), vis_score


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

    def single_marker_ippe_candidates(self,
                                      marker_id: int,
                                      img_pts: np.ndarray,
                                      K: np.ndarray,
                                      D: np.ndarray,
                                      corners_list: Optional[List[np.ndarray]] = None,
                                      ids: Optional[np.ndarray] = None,
                                      depth_u16: Optional[np.ndarray] = None,
                                      depth_scale: Optional[float] = None) -> List[Dict[str, Any]]:
        mid = int(marker_id)
        if not self.model.has_marker(mid):
            return []

        obj_pts = self.model.marker_corners_in_rig(mid).reshape(-1, 1, 3).astype(np.float64)
        img_pts = self.model.reorder_image_corners(mid, np.asarray(img_pts, dtype=np.float64).reshape(4, 2))
        retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            obj_pts,
            img_pts.reshape(-1, 1, 2),
            K,
            D,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if int(retval) <= 0 or len(rvecs) == 0:
            return []

        candidates: List[Dict[str, Any]] = []
        for sol_idx in range(len(rvecs)):
            rvec = np.asarray(rvecs[sol_idx], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvecs[sol_idx], dtype=np.float64).reshape(3, 1)
            tvec_flat = tvec.reshape(3)
            proj2, _ = cv2.projectPoints(obj_pts.reshape(-1, 3), rvec, tvec, K, D)
            proj2 = proj2.reshape(-1, 2)
            err = np.linalg.norm(proj2 - img_pts.reshape(-1, 2), axis=1).astype(np.float64)
            err_mean = float(reproj_errs[sol_idx][0]) if reproj_errs is not None else float(np.mean(err))
            T_C_O = rodrigues_to_Rt(rvec, tvec)
            z_ok = float(tvec_flat[2]) > 0.0
            vis_ok, vis_score = self.model.marker_visibility_score(mid, T_C_O)
            tier = 2
            if z_ok and vis_ok:
                tier = 0
            elif z_ok:
                tier = 1
            depth_metrics = self.score_pose_with_depth(
                depth_u16, depth_scale, K, T_C_O,
                corners_list, ids, only_ids=[mid]
            )
            candidates.append({
                "solution_index": int(sol_idx),
                "rvec": rvec,
                "tvec": tvec,
                "proj2": proj2,
                "err": err,
                "err_mean": float(err_mean),
                "T_C_O": T_C_O,
                "z_ok": bool(z_ok),
                "vis_ok": bool(vis_ok),
                "vis_score": float(vis_score),
                "visibility_tier": int(tier),
                "depth_metrics": depth_metrics,
                "rank": (int(tier), float(err_mean), -float(vis_score)),
            })
        return candidates

    def _median_depth_patch_m(self,
                              depth_u16: np.ndarray,
                              depth_scale: float,
                              u: float,
                              v: float,
                              radius: int = 2) -> Optional[float]:
        h, w = depth_u16.shape[:2]
        x0 = max(0, int(round(u)) - int(radius))
        y0 = max(0, int(round(v)) - int(radius))
        x1 = min(w, int(round(u)) + int(radius) + 1)
        y1 = min(h, int(round(v)) + int(radius) + 1)
        patch = np.asarray(depth_u16[y0:y1, x0:x1], dtype=np.float64)
        if patch.size == 0:
            return None
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        z_m = float(np.median(valid)) * float(depth_scale)
        if not np.isfinite(z_m) or z_m <= 0.0:
            return None
        return z_m

    def get_3d_correspondences_from_depth(self,
                                          bgr: np.ndarray,
                                          depth_u16: np.ndarray,
                                          depth_scale: float,
                                          K: np.ndarray,
                                          only_ids: Optional[List[int]] = None,
                                          min_aspect: float = 0.0,
                                          patch_radius: int = 2,
                                          min_valid_per_marker: int = 3):
        obs_set = self.collect_observations(bgr, only_ids=only_ids, min_aspect=min_aspect)
        if not obs_set.marker_observations:
            return None, None, []

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None, None, []

        obj_pts: List[np.ndarray] = []
        cam_pts: List[List[float]] = []
        used_ids: List[int] = []

        for obs in obs_set.marker_observations:
            valid_obj = []
            valid_cam = []
            for obj_p, img_p in zip(obs.obj_corners_3d, obs.corners_2d_reordered):
                z_m = self._median_depth_patch_m(depth_u16, depth_scale, float(img_p[0]), float(img_p[1]), radius=patch_radius)
                if z_m is None:
                    continue
                x_m = (float(img_p[0]) - cx) * z_m / fx
                y_m = (float(img_p[1]) - cy) * z_m / fy
                valid_obj.append(np.asarray(obj_p, dtype=np.float64))
                valid_cam.append([x_m, y_m, z_m])
            if len(valid_cam) >= int(min_valid_per_marker):
                obj_pts.extend(valid_obj)
                cam_pts.extend(valid_cam)
                used_ids.append(int(obs.marker_id))

        if len(cam_pts) < 3:
            return None, None, []
        return (
            np.asarray(obj_pts, dtype=np.float64),
            np.asarray(cam_pts, dtype=np.float64),
            used_ids,
        )

    @staticmethod
    def _kabsch_svd(obj_pts: np.ndarray, cam_pts: np.ndarray) -> Optional[np.ndarray]:
        if obj_pts is None or cam_pts is None or len(obj_pts) < 3 or len(cam_pts) < 3:
            return None
        obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        cam = np.asarray(cam_pts, dtype=np.float64).reshape(-1, 3)
        if obj.shape != cam.shape:
            return None
        mu_o = np.mean(obj, axis=0)
        mu_c = np.mean(cam, axis=0)
        H = (obj - mu_o).T @ (cam - mu_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_c - R @ mu_o
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def solve_pose_from_depth(self,
                              bgr: np.ndarray,
                              depth_u16: Optional[np.ndarray],
                              depth_scale: Optional[float],
                              K: np.ndarray,
                              D: np.ndarray,
                              only_ids: Optional[List[int]] = None,
                              min_aspect: float = 0.0,
                              patch_radius: int = 2,
                              min_valid_per_marker: int = 3,
                              min_valid_points: int = 6,
                              max_reproj_mean_px: float = 4.0):
        if depth_u16 is None or depth_scale is None:
            return False, None

        obj_pts, cam_pts, used_ids = self.get_3d_correspondences_from_depth(
            bgr,
            depth_u16,
            depth_scale,
            K,
            only_ids=only_ids,
            min_aspect=min_aspect,
            patch_radius=patch_radius,
            min_valid_per_marker=min_valid_per_marker,
        )
        if obj_pts is None or cam_pts is None or len(obj_pts) < int(min_valid_points):
            return False, None

        T_C_O = self._kabsch_svd(obj_pts, cam_pts)
        if T_C_O is None or not np.all(np.isfinite(T_C_O)):
            return False, None
        if float(T_C_O[2, 3]) <= 0.0:
            return False, None

        rvec, _ = cv2.Rodrigues(T_C_O[:3, :3])
        tvec = T_C_O[:3, 3].reshape(3, 1)

        obs_set = self.collect_observations(bgr, only_ids=used_ids, min_aspect=min_aspect)
        if not obs_set.marker_observations:
            return False, None
        reproj_obj = []
        reproj_img = []
        for obs in obs_set.marker_observations:
            reproj_obj.append(np.asarray(obs.obj_corners_3d, dtype=np.float64))
            reproj_img.append(np.asarray(obs.corners_2d_reordered, dtype=np.float64))
        obj_all = np.concatenate(reproj_obj, axis=0).reshape(-1, 1, 3).astype(np.float64)
        img_all = np.concatenate(reproj_img, axis=0).reshape(-1, 1, 2).astype(np.float64)
        proj2, _ = cv2.projectPoints(obj_all.reshape(-1, 3), rvec, tvec, K, D)
        proj2 = proj2.reshape(-1, 2)
        err = np.linalg.norm(proj2 - img_all.reshape(-1, 2), axis=1).astype(np.float64)
        err_mean = float(np.mean(err)) if err.size else float("inf")
        if not np.isfinite(err_mean) or err_mean > float(max_reproj_mean_px):
            return False, None

        corners_list, ids = self.detect(bgr)
        depth_metrics = self.score_pose_with_depth(
            depth_u16,
            depth_scale,
            K,
            T_C_O,
            corners_list,
            ids,
            only_ids=used_ids,
        )
        return True, {
            "T_C_O": T_C_O,
            "rvec": rvec,
            "tvec": tvec,
            "err": err,
            "err_mean": err_mean,
            "proj2": proj2,
            "used_ids": [int(x) for x in used_ids],
            "n_points": int(err.size),
            "source": "depth_svd",
            "depth_metrics": depth_metrics,
        }

    @staticmethod
    def _shrink_quad(quad_xy: np.ndarray, scale: float) -> np.ndarray:
        quad = np.asarray(quad_xy, dtype=np.float64).reshape(4, 2)
        center = np.mean(quad, axis=0, keepdims=True)
        return center + (quad - center) * float(scale)

    @staticmethod
    def _median_depth_at(depth_u16: np.ndarray, u: int, v: int, patch_radius: int = 1) -> Optional[int]:
        h, w = depth_u16.shape[:2]
        x0 = max(0, int(u) - int(patch_radius))
        y0 = max(0, int(v) - int(patch_radius))
        x1 = min(w, int(u) + int(patch_radius) + 1)
        y1 = min(h, int(v) + int(patch_radius) + 1)
        patch = depth_u16[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return int(np.median(valid))

    def score_pose_with_depth(self,
                              depth_u16: Optional[np.ndarray],
                              depth_scale: Optional[float],
                              K: np.ndarray,
                              T_C_O: np.ndarray,
                              corners_list: Optional[List[np.ndarray]],
                              ids: Optional[np.ndarray],
                              only_ids: Optional[List[int]] = None,
                              quad_shrink: float = 0.7,
                              stride_px: int = 4,
                              patch_radius: int = 1,
                              min_samples: int = 8) -> Dict[str, Any]:
        metrics = {
            "valid": False,
            "num_samples": 0,
            "num_markers": 0,
            "support_marker_ids": [],
            "plane_mean_mm": None,
            "plane_median_mm": None,
            "plane_max_mm": None,
            "z_mean_mm": None,
            "z_median_mm": None,
            "inlier_ratio": None,
        }
        if depth_u16 is None or depth_scale is None or corners_list is None or ids is None:
            return metrics

        depth = np.asarray(depth_u16)
        if depth.ndim != 2 or depth.size == 0:
            return metrics

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return metrics

        only_set = set(int(x) for x in only_ids) if only_ids is not None else None
        plane_errs = []
        z_errs = []
        support_marker_ids: List[int] = []
        h, w = depth.shape[:2]

        for corners, mid in zip(corners_list, ids):
            mid = int(mid)
            if not self.model.has_marker(mid):
                continue
            if only_set is not None and mid not in only_set:
                continue

            quad = self._shrink_quad(np.asarray(corners, dtype=np.float64).reshape(4, 2), quad_shrink)
            x0 = max(0, int(np.floor(np.min(quad[:, 0]))))
            y0 = max(0, int(np.floor(np.min(quad[:, 1]))))
            x1 = min(w, int(np.ceil(np.max(quad[:, 0]))))
            y1 = min(h, int(np.ceil(np.max(quad[:, 1]))))
            if x1 <= x0 or y1 <= y0:
                continue

            T_C_M = np.asarray(T_C_O, dtype=np.float64) @ self.model.marker_pose_in_rig(mid)
            plane_origin = np.asarray(T_C_M[:3, 3], dtype=np.float64)
            plane_normal = np.asarray(T_C_M[:3, 2], dtype=np.float64)
            normal_norm = np.linalg.norm(plane_normal)
            if normal_norm < 1e-9:
                continue
            plane_normal = plane_normal / normal_norm

            marker_samples_before = len(plane_errs)
            poly = quad.astype(np.float32)
            for v in range(y0, y1, max(int(stride_px), 1)):
                for u in range(x0, x1, max(int(stride_px), 1)):
                    inside = cv2.pointPolygonTest(poly, (float(u), float(v)), False)
                    if inside < 0:
                        continue

                    depth_raw = self._median_depth_at(depth, u, v, patch_radius=patch_radius)
                    if depth_raw is None:
                        continue
                    z_meas = float(depth_raw) * float(depth_scale)
                    if not np.isfinite(z_meas) or z_meas <= 0.0:
                        continue

                    ray = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
                    denom = float(np.dot(plane_normal, ray))
                    if abs(denom) < 1e-9:
                        continue
                    z_pred = float(np.dot(plane_normal, plane_origin) / denom)
                    if not np.isfinite(z_pred) or z_pred <= 0.0:
                        continue

                    P_meas = ray * z_meas
                    plane_errs.append(abs(float(np.dot(plane_normal, P_meas - plane_origin))) * 1000.0)
                    z_errs.append(abs(z_meas - z_pred) * 1000.0)

            if len(plane_errs) > marker_samples_before:
                support_marker_ids.append(mid)

        if not plane_errs:
            return metrics

        plane_arr = np.asarray(plane_errs, dtype=np.float64)
        z_arr = np.asarray(z_errs, dtype=np.float64)
        metrics.update({
            "valid": bool(len(plane_arr) >= int(min_samples)),
            "num_samples": int(len(plane_arr)),
            "num_markers": int(len(support_marker_ids)),
            "support_marker_ids": [int(x) for x in support_marker_ids],
            "plane_mean_mm": float(np.mean(plane_arr)),
            "plane_median_mm": float(np.median(plane_arr)),
            "plane_max_mm": float(np.max(plane_arr)),
            "z_mean_mm": float(np.mean(z_arr)),
            "z_median_mm": float(np.median(z_arr)),
            "inlier_ratio": float(np.mean(plane_arr <= 8.0)),
        })
        return metrics

    def collect_observations(self, bgr: np.ndarray,
                             only_ids: Optional[List[int]] = None,
                             min_aspect: float = 0.0) -> CubeObservationSet:
        corners_list, ids = self.detect(bgr)
        only_set = set(int(x) for x in only_ids) if only_ids is not None else None
        observations: List[MarkerObservation] = []
        if ids is None:
            return CubeObservationSet([])

        for corners, mid in zip(corners_list, ids):
            mid = int(mid)
            if not self.model.has_marker(mid):
                continue
            if only_set is not None and mid not in only_set:
                continue

            raw_img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            img = self.model.reorder_image_corners(mid, raw_img)
            edge_w = float(np.linalg.norm(img[1] - img[0]))
            edge_h = float(np.linalg.norm(img[3] - img[0]))
            aspect = min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6)
            if min_aspect > 0 and aspect < float(min_aspect):
                continue

            observations.append(MarkerObservation(
                marker_id=mid,
                face_name=self.model.marker_face_name(mid),
                corners_2d=raw_img,
                corners_2d_reordered=img,
                obj_corners_3d=self.model.marker_corners_in_rig(mid),
                aspect_ratio=float(aspect),
                T_object_marker=self.model.marker_pose_in_rig(mid),
            ))

        return CubeObservationSet(observations)

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
            if not self.model.has_marker(mid):
                continue
            if only_set is not None and mid not in only_set:
                continue

            obj = self.model.marker_corners_in_rig(mid)
            img = self.model.reorder_image_corners(mid, c.reshape(4, 2))

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
                       min_aspect: float = 0.3,
                       depth_u16: Optional[np.ndarray] = None,
                       depth_scale: Optional[float] = None):
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

        if n == 4:
            marker_id = int(used[0]) if len(used) == 1 else None
            if marker_id is None:
                return (False, None, None, used, None) if return_reproj else (False, None, None, used)

            candidates = self.single_marker_ippe_candidates(
                marker_id,
                img_pts.reshape(-1, 2),
                K,
                D,
                corners_list=corners_list,
                ids=ids,
                depth_u16=depth_u16,
                depth_scale=depth_scale,
            )
            if not candidates:
                return (False, None, None, used, None) if return_reproj else (False, None, None, used)
            best = min(candidates, key=lambda cand: cand["rank"])

            reproj = {
                "obj_pts": obj_pts,
                "img_pts": img_pts,
                "proj2": best["proj2"],
                "err": best["err"],
                "err_mean": float(best["err_mean"]),
                "err_median": float(np.median(best["err"])),
                "err_p90": float(np.percentile(best["err"], 90)),
                "n_points": int(best["err"].size),
                "rvec": best["rvec"],
                "tvec": best["tvec"],
                "depth_metrics": best["depth_metrics"],
                "solution_index": int(best["solution_index"]),
                "z_ok": bool(best["z_ok"]),
                "vis_ok": bool(best["vis_ok"]),
                "vis_score": float(best["vis_score"]),
                "visibility_tier": int(best["visibility_tier"]),
            }
            ok_final = reproj["err_mean"] <= reproj_thr_mean_px
            if return_reproj:
                return ok_final, best["rvec"], best["tvec"], used, reproj
            return ok_final, best["rvec"], best["tvec"], used

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
        if not np.all(np.isfinite(err)):
            return (False, None, None, used, None) if return_reproj else (False, None, None, used)

        reproj = {
            "obj_pts": obj_pts, "img_pts": img_pts, "proj2": proj2, "err": err,
            "err_mean": float(np.mean(err)), "err_median": float(np.median(err)),
            "err_p90": float(np.percentile(err, 90)), "n_points": int(err.size),
            "rvec": rvec, "tvec": tvec,
        }
        if depth_u16 is not None and depth_scale is not None:
            reproj["depth_metrics"] = self.score_pose_with_depth(
                depth_u16, depth_scale, K,
                rodrigues_to_Rt(rvec, tvec),
                corners_list, ids, only_ids=used)
        if not np.isfinite(reproj["err_mean"]):
            return (False, None, None, used, None) if return_reproj else (False, None, None, used)
        ok_final = reproj["err_mean"] <= reproj_thr_mean_px

        if return_reproj:
            return ok_final, rvec, tvec, used, reproj
        return ok_final, rvec, tvec, used
