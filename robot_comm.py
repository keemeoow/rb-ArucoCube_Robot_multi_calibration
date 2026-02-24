# robot_comm.py
"""
Robot communication module.
Sends joint commands and receives robot TCP poses over socket.
"""

import socket
import json
import time
import numpy as np
from typing import Optional, Tuple, List


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


class RobotClient:
    """Socket client that communicates with the robot server (Zeus-style protocol)."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"[RobotClient] Connected to {self.host}:{self.port}")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def wait_for_command(self) -> str:
        """Wait for a command from the server (e.g., 'capture', 'quit')."""
        data = self.sock.recv(4096).decode('utf-8').strip()
        return data

    def send_joint_command(self, joints: List[float]):
        """Send a joint pose to the robot server."""
        d1, d2, d3, d4, d5, d6 = joints
        result = {
            'status': 'success',
            'action': 'capture',
            'd1': d1, 'd2': d2, 'd3': d3,
            'd4': d4, 'd5': d5, 'd6': d6
        }
        self.sock.sendall(json.dumps(result).encode('utf-8'))

    def send_pose_and_wait(self, joints: List[float], settle_time: float = 1.5) -> bool:
        """
        Wait for 'capture' command, send joints, wait for settle.
        Returns True on success, False on quit/error.
        """
        try:
            cmd = self.wait_for_command()
            if cmd == 'quit':
                return False
            if cmd == 'capture':
                self.send_joint_command(joints)
                time.sleep(settle_time)
                return True
            print(f"[RobotClient] Unknown command: {cmd}")
            return False
        except Exception as e:
            print(f"[RobotClient] Error: {e}")
            return False
