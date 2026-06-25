# robot_comm.py
"""
Robot pose helpers for the calibration capture workflow.

euler_deg_to_matrix(): convert a robot 6-DoF pose (x,y,z in mm, rz,ry,rx in deg)
to a 4x4 homogeneous transform (translation in meters). Used by Step2_capture.py
and the Step3 calibration to turn robot TCP poses into matrices.

The live capture protocol itself (request_waypoints / capture / save_waypoints,
newline-delimited JSON) is implemented directly with a raw socket in
Step2_capture.py's manual-robot mode against server/robot_calb.py.
"""

import numpy as np


def euler_deg_to_matrix(x_mm, y_mm, z_mm, rz_deg, ry_deg, rx_deg) -> np.ndarray:
    """
    Convert robot pose (x,y,z in mm, rz,ry,rx in deg) to 4x4 homogeneous matrix.
    Convention: ZYX extrinsic (Rz @ Ry @ Rx), translation in meters.
    """
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz),  np.cos(rz), 0],
                    [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx),  np.cos(rx)]], dtype=np.float64)
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
