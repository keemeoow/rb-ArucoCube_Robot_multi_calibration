# robot_comm.py
"""
Robot communication module for Place-and-Capture calibration workflow.

Protocol:
  1. Server -> Client: {"command": "ready", "pose_index": N}
  2. Client -> Server: {"action": "waypoint", "place_pose": [6], "capture_pose": [6]}
                    or {"action": "quit"}
  3. Server executes place-capture-pickup cycle
  4. Server -> Client: {"command": "capture", "capture_pose_6dof": [...], "place_pose_6dof": [...]}
  5. Client -> Server: {"action": "captured", "status": "success"}
  6. Repeat from 1
"""

import socket
import json
import time
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


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


class PlaceCaptureClient:
    """Socket client for the Place-and-Capture calibration protocol."""

    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"[PlaceCaptureClient] Connected to {self.host}:{self.port}")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _send_json(self, obj: dict):
        msg = json.dumps(obj)
        self.sock.sendall(msg.encode("utf-8"))

    def _recv_json(self) -> dict:
        data = self.sock.recv(8192)
        if not data:
            raise RuntimeError("socket closed by peer")
        txt = data.decode("utf-8").strip()
        return json.loads(txt)

    def wait_for_ready(self) -> Dict[str, Any]:
        """
        Wait for server 'ready' signal.
        Returns: {"command": "ready", "pose_index": N}
        """
        msg = self._recv_json()
        cmd = msg.get("command", "")
        if cmd == "quit":
            raise RuntimeError("Server sent quit")
        if cmd != "ready":
            print(f"[PlaceCaptureClient] Expected 'ready', got '{cmd}'")
        return msg

    def send_waypoint(self,
                      place_pose: List[float],
                      capture_pose: List[float],
                      place_kind: Optional[str] = None,
                      capture_kind: Optional[str] = None,
                      extra_fields: Optional[Dict[str, Any]] = None):
        """Send place + capture waypoint pair to server."""
        payload = {
            "action": "waypoint",
            "place_pose": [float(x) for x in place_pose],
            "capture_pose": [float(x) for x in capture_pose],
        }
        if place_kind:
            payload["place_pose_kind"] = str(place_kind)
        if capture_kind:
            payload["capture_pose_kind"] = str(capture_kind)
        if extra_fields:
            payload.update(extra_fields)
        self._send_json(payload)

    def send_quit(self):
        """Tell server to quit."""
        self._send_json({"action": "quit"})

    def wait_for_capture_signal(self) -> Dict[str, Any]:
        """
        Wait for server 'capture' signal after robot has placed cube and moved up.
        Returns: {
            "command": "capture",
            "capture_pose_6dof": [...],
            "place_pose_6dof": [...],
            "pose_index": N
        }
        """
        msg = self._recv_json()
        cmd = msg.get("command", "")
        if cmd == "quit":
            raise RuntimeError("Server sent quit")
        if cmd != "capture":
            print(f"[PlaceCaptureClient] Expected 'capture', got '{cmd}'")
        return msg

    def send_captured(self, status: str = "success", reason: Optional[str] = None):
        """Acknowledge capture completion to server."""
        payload = {
            "action": "captured",
            "status": status,
        }
        if reason:
            payload["reason"] = str(reason)
        self._send_json(payload)

    def run_single_waypoint(
        self,
        place_pose: List[float],
        capture_pose: List[float],
        place_kind: Optional[str] = None,
        capture_kind: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[List[float]], Optional[List[float]]]:
        """
        Full cycle for one waypoint:
          1. Wait for 'ready'
          2. Send waypoint
          3. Wait for 'capture' signal
        Returns: (ok, capture_pose_6dof, place_pose_6dof)
        """
        try:
            self.wait_for_ready()
            self.send_waypoint(
                place_pose,
                capture_pose,
                place_kind=place_kind,
                capture_kind=capture_kind,
                extra_fields=extra_fields,
            )
            cap_msg = self.wait_for_capture_signal()

            capture_tcp = cap_msg.get("capture_pose_6dof")
            place_tcp = cap_msg.get("place_pose_6dof")
            return True, capture_tcp, place_tcp
        except Exception as e:
            print(f"[PlaceCaptureClient] Error: {e}")
            return False, None, None

    def acknowledge_capture(self):
        """Send capture acknowledgement after saving images."""
        self.send_captured("success")
