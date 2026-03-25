# config.py
"""
캘리브레이션 파이프라인 공통 설정.
실제 하드웨어에 맞게 값을 수정하여 사용.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CubeConfig:
    """ArUco cube target configuration."""
    cube_side_m: float = 0.03          # cube edge length (m) - 30mm
    marker_size_m: float = 0.022       # marker size on each face (m) - 22mm
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # marker_id -> face name
    id_to_face: Dict[int, str] = field(default_factory=lambda: {
        0: "+Y",
        1: "+Z",
        2: "+X",
        3: "-Z",
        4: "-X",
    })

    # per-marker in-plane rotation (deg) if physically rotated
    face_roll_deg: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0
    })


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
