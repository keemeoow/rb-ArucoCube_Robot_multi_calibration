import copy
import itertools
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CubeConfig, get_default_cube_config


ALL_CORNER_PERMUTATIONS = tuple(tuple(p) for p in itertools.permutations(range(4)))
LOCKED_CORNER_REORDER_MARKERS = {3}


def clone_cube_config(cfg: CubeConfig) -> CubeConfig:
    return copy.deepcopy(cfg)


def cube_config_to_dict(cfg: CubeConfig) -> dict:
    return {
        "cube_side_m": float(cfg.cube_side_m),
        "marker_size_m": float(cfg.marker_size_m),
        "dictionary_name": str(cfg.dictionary_name),
        "marker_ids": [int(x) for x in cfg.marker_ids],
        "id_to_face": {str(k): str(v) for k, v in cfg.id_to_face.items()},
        "corner_reorder": {str(k): [int(x) for x in v] for k, v in cfg.corner_reorder.items()},
        "face_roll_deg": {str(k): float(v) for k, v in cfg.face_roll_deg.items()},
        "marker_pose_4x4": {
            str(k): [[float(x) for x in row] for row in np.asarray(v, dtype=np.float64).reshape(4, 4)]
            for k, v in cfg.marker_pose_4x4.items()
        },
    }


def _normalize_cube_config_value(value):
    if isinstance(value, dict):
        return {
            str(k): _normalize_cube_config_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_cube_config_value(v) for v in value]
    if isinstance(value, float):
        if abs(value) < 1e-12:
            value = 0.0
        return round(float(value), 9)
    return value


def cube_config_to_comparable_dict(cfg: CubeConfig) -> dict:
    return _normalize_cube_config_value(cube_config_to_dict(cfg))


def cube_configs_equivalent(a: CubeConfig, b: CubeConfig) -> bool:
    return cube_config_to_comparable_dict(a) == cube_config_to_comparable_dict(b)


def cube_config_mismatch_keys(expected: CubeConfig, actual: CubeConfig) -> List[str]:
    exp = cube_config_to_comparable_dict(expected)
    act = cube_config_to_comparable_dict(actual)
    keys = [
        "cube_side_m",
        "marker_size_m",
        "dictionary_name",
        "marker_ids",
        "id_to_face",
        "corner_reorder",
        "face_roll_deg",
        "marker_pose_4x4",
    ]
    return [key for key in keys if exp.get(key) != act.get(key)]


def cube_config_from_dict(data: dict, base_cfg: Optional[CubeConfig] = None) -> CubeConfig:
    cfg = clone_cube_config(base_cfg or get_default_cube_config())
    cfg.cube_side_m = float(data.get("cube_side_m", cfg.cube_side_m))
    cfg.marker_size_m = float(data.get("marker_size_m", cfg.marker_size_m))
    cfg.dictionary_name = str(data.get("dictionary_name", cfg.dictionary_name))
    cfg.marker_ids = tuple(int(x) for x in data.get("marker_ids", list(cfg.marker_ids)))
    if "id_to_face" in data:
        cfg.id_to_face = {int(k): str(v) for k, v in data["id_to_face"].items()}
    if "corner_reorder" in data:
        cfg.corner_reorder = {int(k): [int(x) for x in v] for k, v in data["corner_reorder"].items()}
    if "face_roll_deg" in data:
        cfg.face_roll_deg = {int(k): float(v) for k, v in data["face_roll_deg"].items()}
    if "marker_pose_4x4" in data:
        cfg.marker_pose_4x4 = {
            int(k): np.asarray(v, dtype=np.float64).reshape(4, 4).tolist()
            for k, v in data["marker_pose_4x4"].items()
        }
    return cfg


def cube_config_from_search_result(data: dict, base_cfg: Optional[CubeConfig] = None) -> CubeConfig:
    cfg = clone_cube_config(base_cfg or get_default_cube_config())
    rec = data.get("recommended", data)
    if "base_cube_config" in data and isinstance(data["base_cube_config"], dict):
        cfg = cube_config_from_dict(data["base_cube_config"], cfg)
    elif "initial" in data and isinstance(data["initial"], dict):
        base_rec = data["initial"].get("base_cube_config")
        if isinstance(base_rec, dict):
            cfg = cube_config_from_dict(base_rec, cfg)

    if "id_to_face" in rec:
        cfg.id_to_face = {int(k): str(v) for k, v in rec["id_to_face"].items()}
    if "corner_reorder" in rec:
        cfg.corner_reorder = {int(k): [int(x) for x in v] for k, v in rec["corner_reorder"].items()}
    if "face_roll_deg" in rec:
        cfg.face_roll_deg = {int(k): float(v) for k, v in rec["face_roll_deg"].items()}
    if "marker_pose_4x4" in rec:
        cfg.marker_pose_4x4 = {
            int(k): np.asarray(v, dtype=np.float64).reshape(4, 4).tolist()
            for k, v in rec["marker_pose_4x4"].items()
        }
    return cfg


def load_cube_config_from_json_file(path: str,
                                    default_cfg: Optional[CubeConfig] = None) -> Tuple[Optional[CubeConfig], str]:
    if not path or not os.path.exists(path):
        return None, "missing"
    cfg = clone_cube_config(default_cfg or get_default_cube_config())
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, "missing"

    try:
        if isinstance(data, dict) and ("recommended" in data or "initial" in data):
            return cube_config_from_search_result(data, cfg), "json_search_result"
        if isinstance(data, dict):
            return cube_config_from_dict(data, cfg), "json_config"
    except Exception:
        return None, "missing"
    return None, "missing"


def infer_cube_config_from_legacy_meta(root_folder: str,
                                       default_cfg: Optional[CubeConfig] = None,
                                       meta: Optional[dict] = None) -> Tuple[Optional[CubeConfig], dict]:
    cfg = clone_cube_config(default_cfg or get_default_cube_config())
    marker_ids = {int(x) for x in cfg.marker_ids}
    meta_path = os.path.join(root_folder, "meta.json")

    if meta is None:
        if not os.path.exists(meta_path):
            return None, {"reason": "missing_meta"}
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            return None, {"reason": "invalid_meta"}

    captures = meta.get("captures")
    if not isinstance(captures, list) or not captures:
        return None, {"reason": "no_legacy_captures"}

    face_votes: Dict[int, Counter] = defaultdict(Counter)
    perm_votes: Dict[int, Counter] = defaultdict(Counter)
    perm_error_sums: Dict[int, Dict[Tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))

    try:
        dictionary_id = getattr(cv2.aruco, cfg.dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        try:
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

            def detect(gray):
                corners, ids, _ = detector.detectMarkers(gray)
                return corners, ids
        except AttributeError:
            detector_params = cv2.aruco.DetectorParameters_create()

            def detect(gray):
                corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
                return corners, ids
    except AttributeError:
        return None, {"reason": "invalid_dictionary", "dictionary_name": cfg.dictionary_name}

    for cap in captures:
        for cinfo in cap.get("cams", {}).values():
            markers = cinfo.get("markers", [])
            if not markers:
                continue

            rgb_rel = cinfo.get("rgb_path", "")
            raw_by_id = {}
            if rgb_rel:
                img = cv2.imread(os.path.join(root_folder, rgb_rel))
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    det_corners, det_ids = detect(gray)
                    if det_ids is not None:
                        raw_by_id = {
                            int(mid): corners.reshape(4, 2).astype(np.float64)
                            for corners, mid in zip(det_corners, det_ids.flatten().astype(int))
                        }

            for marker in markers:
                raw_mid = marker.get("marker_id", marker.get("id", -1))
                try:
                    mid = int(raw_mid)
                except (TypeError, ValueError):
                    continue
                if mid not in marker_ids:
                    continue

                face = marker.get("face")
                if isinstance(face, str) and face:
                    face_votes[mid][face] += 1

                stored = np.asarray(marker.get("corners_2d", []), dtype=np.float64)
                raw = raw_by_id.get(mid)
                if raw is None or stored.shape != (4, 2):
                    continue

                best_perm = None
                best_err = float("inf")
                for perm in ALL_CORNER_PERMUTATIONS:
                    err = float(np.linalg.norm(raw[list(perm)] - stored, axis=1).sum())
                    if err < best_err:
                        best_err = err
                        best_perm = perm
                if best_perm is None:
                    continue
                perm_votes[mid][best_perm] += 1
                perm_error_sums[mid][best_perm] += best_err

    inferred = clone_cube_config(cfg)
    face_report = {}
    perm_report = {}
    changed = False

    for mid in inferred.marker_ids:
        mid = int(mid)
        current_face = inferred.id_to_face.get(mid)
        current_perm = tuple(inferred.corner_reorder.get(mid, [0, 1, 2, 3]))

        if face_votes[mid]:
            best_face, best_face_votes = sorted(
                face_votes[mid].items(), key=lambda item: (-item[1], item[0]))[0]
            inferred.id_to_face[mid] = best_face
            changed = changed or (best_face != current_face)
            face_report[mid] = {
                "selected": best_face,
                "votes": dict(face_votes[mid]),
                "vote_count": int(best_face_votes),
            }
        else:
            face_report[mid] = {
                "selected": current_face,
                "votes": {},
                "vote_count": 0,
            }

        if mid in LOCKED_CORNER_REORDER_MARKERS:
            perm_report[mid] = {
                "selected": list(current_perm),
                "vote_count": 0 if not perm_votes[mid] else int(sum(perm_votes[mid].values())),
                "avg_corner_error_px": None,
                "votes": {
                    json.dumps(list(perm)): int(vote_count)
                    for perm, vote_count in perm_votes[mid].items()
                },
                "locked_to_default": True,
            }
        elif perm_votes[mid]:
            best_perm, best_perm_votes = sorted(
                perm_votes[mid].items(),
                key=lambda item: (
                    -item[1],
                    perm_error_sums[mid][item[0]] / max(item[1], 1),
                    item[0],
                ),
            )[0]
            inferred.corner_reorder[mid] = list(best_perm)
            changed = changed or (tuple(best_perm) != current_perm)
            perm_report[mid] = {
                "selected": list(best_perm),
                "vote_count": int(best_perm_votes),
                "avg_corner_error_px": (
                    perm_error_sums[mid][best_perm] / max(best_perm_votes, 1)
                ),
                "votes": {
                    json.dumps(list(perm)): int(vote_count)
                    for perm, vote_count in perm_votes[mid].items()
                },
            }
        else:
            perm_report[mid] = {
                "selected": list(current_perm),
                "vote_count": 0,
                "avg_corner_error_px": None,
                "votes": {},
                "locked_to_default": mid in LOCKED_CORNER_REORDER_MARKERS,
            }

    info = {
        "reason": "ok",
        "face_votes": {str(mid): face_report[int(mid)] for mid in inferred.marker_ids},
        "corner_votes": {str(mid): perm_report[int(mid)] for mid in inferred.marker_ids},
        "changed_from_default": bool(changed),
    }
    if not changed:
        return None, info
    return inferred, info


def load_cube_config_from_meta(root_folder: str, default_cfg: Optional[CubeConfig] = None) -> Tuple[CubeConfig, str]:
    cfg = clone_cube_config(default_cfg or get_default_cube_config())
    meta_path = os.path.join(root_folder, "meta.json")
    if not os.path.exists(meta_path):
        return cfg, "default"

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return cfg, "default"

    cube_data = meta.get("cube_config")
    if isinstance(cube_data, dict):
        try:
            return cube_config_from_dict(cube_data, cfg), "meta"
        except Exception:
            pass

    legacy_cfg, _ = infer_cube_config_from_legacy_meta(root_folder, cfg, meta=meta)
    if legacy_cfg is not None:
        return legacy_cfg, "legacy_meta"
    return cfg, "default"


def load_cube_config_from_calibration_summary(calib_dir: str,
                                              default_cfg: Optional[CubeConfig] = None) -> Tuple[Optional[CubeConfig], str]:
    cfg = clone_cube_config(default_cfg or get_default_cube_config())
    summary_path = os.path.join(calib_dir, "calibration_summary.json")
    if not os.path.exists(summary_path):
        return None, "missing"

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception:
        return None, "missing"

    cube_data = summary.get("cube_config_used")
    if not isinstance(cube_data, dict):
        return None, "missing"

    try:
        return cube_config_from_dict(cube_data, cfg), str(summary.get("cube_config_source", "calibration_summary"))
    except Exception:
        return None, "missing"
