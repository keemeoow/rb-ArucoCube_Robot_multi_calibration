# config.py
"""
캘리브레이션 파이프라인 공통 설정.

이 파일의 기본 CubeConfig는 현재 프로젝트에서 계속 재사용하는 단일 기준 큐브 정의다.
동일한 물리 큐브를 계속 사용할 경우, 별도 override 없이 이 정의를 그대로 사용한다.
예외적으로 다른 큐브/실험 정의가 필요할 때만 명시적인 JSON override를 사용한다.
"""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


_CANONICAL_CUBE_CONFIG_FALLBACK_DATA = {
    "cube_side_m": 0.03,
    "marker_size_m": 0.022,
    "dictionary_name": "DICT_4X4_50",
    "marker_ids": [0, 1, 2, 3, 4],
    "id_to_face": {
        "0": "+Z",
        "1": "+X",
        "2": "+Y",
        "3": "-X",
        "4": "-Y",
    },
    "corner_reorder": {
        "0": [0, 1, 2, 3],
        "1": [0, 1, 2, 3],
        "2": [0, 1, 2, 3],
        "3": [0, 1, 2, 3],
        "4": [0, 1, 2, 3],
    },
    "face_roll_deg": {
        "0": 0.0,
        "1": 270.0,
        "2": 0.0,
        "3": 90.0,
        "4": 180.0,
    },
    "marker_pose_4x4": {},
}


@dataclass
class CubeConfig:
    """ArUco cube target configuration."""
    cube_side_m: float = 0.03          # cube edge length (m) - 30mm
    marker_size_m: float = 0.022       # marker size on each face (m) - 22mm
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # marker_id -> face name
    # Cube net:
    #         [ID0=+Z]
    #   [ID1=+X][ID2=+Y][ID3=-X][ID4=-Y]
    id_to_face: Dict[int, str] = field(default_factory=lambda: {
        0: "+Z",
        1: "+X",
        2: "+Y",
        3: "-X",
        4: "-Y",
    })

    # The validated cube definition uses a single shared local-corner convention,
    # so image corners are consumed in detector order by default.
    corner_reorder: Dict[int, list] = field(default_factory=lambda: {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3],
        3: [0, 1, 2, 3],
        4: [0, 1, 2, 3],
    })

    # per-marker in-plane rotation (deg) validated against the physical cube
    face_roll_deg: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0, 1: 270.0, 2: 0.0, 3: 90.0, 4: 180.0
    })

    # Optional explicit rigid pose of each marker in the cube/object frame.
    # When present, this overrides face-based geometry construction for that
    # marker. corner_reorder is still used to map detector corners to the
    # marker's local [0,1,2,3] order.
    marker_pose_4x4: Dict[int, list] = field(default_factory=dict)


def get_default_cube_config_json_path() -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, "configs", "cube_models", "cube_model_hybrid_v1.json")


def _load_canonical_cube_config_data() -> Tuple[dict, str]:
    path = get_default_cube_config_json_path()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data, f"project_json_default:{path}"
    except Exception:
        pass
    return copy.deepcopy(_CANONICAL_CUBE_CONFIG_FALLBACK_DATA), "config_py_fallback"


def get_default_cube_config() -> CubeConfig:
    """Return a fresh copy of the validated reusable cube definition."""
    data, _ = _load_canonical_cube_config_data()
    data = copy.deepcopy(data)
    return CubeConfig(
        cube_side_m=float(data["cube_side_m"]),
        marker_size_m=float(data["marker_size_m"]),
        dictionary_name=str(data["dictionary_name"]),
        marker_ids=tuple(int(x) for x in data["marker_ids"]),
        id_to_face={int(k): str(v) for k, v in data["id_to_face"].items()},
        corner_reorder={int(k): [int(x) for x in v] for k, v in data["corner_reorder"].items()},
        face_roll_deg={int(k): float(v) for k, v in data["face_roll_deg"].items()},
        marker_pose_4x4={
            int(k): [[float(x) for x in row] for row in value]
            for k, value in data["marker_pose_4x4"].items()
        },
    )


def get_default_cube_config_source() -> str:
    _, source = _load_canonical_cube_config_data()
    return source


@dataclass
class CharucoBoardConfig:
    """ChArUco board target configuration (for eye-in-hand / gripper camera)."""
    squares_x: int = 11           # number of squares in X
    squares_y: int = 7            # number of squares in Y
    square_length_m: float = 0.025   # checker square side (m) - 25mm
    marker_length_m: float = 0.018   # ArUco marker side (m) - 18mm
    dictionary_name: str = "DICT_4X4_250"  # 7x11 board needs ~39 markers
    marker_id_start: int = 5      # reserve cube IDs 0~4


@dataclass
class CameraStreamConfig:
    """RealSense stream config."""
    color_w: int = 640
    color_h: int = 480
    depth_w: int = 640
    depth_h: int = 480
    fps: int = 15


@dataclass
class RobotConfig:
    """Robot communication config."""
    host: str = "192.168.0.23"
    port: int = 12348
    # Euler convention for your robot (ZYX intrinsic = extrinsic XYZ)
    # robot_poses format: [x_mm, y_mm, z_mm, rz_deg, ry_deg, rx_deg]
    euler_order: str = "ZYX"


@dataclass
class CalibrationConfig:
    """Calibration parameters."""
    # ArUco detection
    min_markers: int = 1
    reproj_max_px: float = 10.0
    use_ransac: bool = True

    # Hand-eye
    handeye_method: int = 4   # cv2.CALIB_HAND_EYE_PARK

    # Multi-cam
    ref_fixed_cam_idx: int = 1       # which fixed camera is the reference
    gripper_cam_idx: int = 0         # which cam index is the gripper camera

    # Point cloud fusion
    z_min: float = 0.2
    z_max: float = 1.5
    stride: int = 4
