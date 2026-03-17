# robot_comm.py
"""
Robot communication module.
Sends joint commands and receives robot TCP poses over socket.
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

    @staticmethod
    def _to_float_pose6(vals) -> Optional[List[float]]:
        try:
            arr = [float(x) for x in vals]
        except Exception:
            return None
        if len(arr) != 6:
            return None
        return arr

    @classmethod
    def _extract_pose6_from_obj(cls, obj: Any) -> Optional[List[float]]:
        """
        Normalize many possible Zeus payload formats into:
          [x_mm, y_mm, z_mm, rz_deg, ry_deg, rx_deg]
        """
        if obj is None:
            return None

        if isinstance(obj, list):
            return cls._to_float_pose6(obj)

        if not isinstance(obj, dict):
            return None

        for k in ["robot_pose_6dof", "tcp_pose_6dof", "pose_6dof", "tcp_pose", "pose"]:
            if k in obj:
                p = cls._extract_pose6_from_obj(obj[k])
                if p is not None:
                    return p

        xyz_rzryrx_keys = ["x", "y", "z", "rz", "ry", "rx"]
        if all(k in obj for k in xyz_rzryrx_keys):
            try:
                return [
                    float(obj["x"]), float(obj["y"]), float(obj["z"]),
                    float(obj["rz"]), float(obj["ry"]), float(obj["rx"]),
                ]
            except Exception:
                pass

        xyz_rpy_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        if all(k in obj for k in xyz_rpy_keys):
            try:
                rx = float(obj["roll"])
                ry = float(obj["pitch"])
                rz = float(obj["yaw"])
                return [float(obj["x"]), float(obj["y"]), float(obj["z"]), rz, ry, rx]
            except Exception:
                pass

        pos = obj.get("position")
        ori = obj.get("orientation")
        if isinstance(pos, dict) and isinstance(ori, dict):
            if all(k in pos for k in ["x", "y", "z"]) and all(k in ori for k in ["rz", "ry", "rx"]):
                try:
                    return [
                        float(pos["x"]), float(pos["y"]), float(pos["z"]),
                        float(ori["rz"]), float(ori["ry"]), float(ori["rx"]),
                    ]
                except Exception:
                    pass
            if all(k in pos for k in ["x", "y", "z"]) and all(k in ori for k in ["yaw", "pitch", "roll"]):
                try:
                    return [
                        float(pos["x"]), float(pos["y"]), float(pos["z"]),
                        float(ori["yaw"]), float(ori["pitch"]), float(ori["roll"]),
                    ]
                except Exception:
                    pass

        for k in ["data", "payload", "robot", "tcp", "state"]:
            if k in obj:
                p = cls._extract_pose6_from_obj(obj[k])
                if p is not None:
                    return p
        return None

    def _recv_text(self) -> str:
        data = self.sock.recv(4096)
        if not data:
            raise RuntimeError("socket closed by peer")
        return data.decode("utf-8").strip()

    def wait_for_command(self) -> str:
        """Wait for a command from the server (e.g., 'capture', 'quit')."""
        return self._recv_text()

    def wait_for_command_packet(self) -> Dict[str, Any]:
        """
        Wait for a command packet and normalize to:
          {
            "command": "capture"/"quit"/..., 
            "tcp_pose_6dof": Optional[List[float]],
            "raw": original decoded object or string
          }
        """
        txt = self._recv_text()
        raw: Any = txt
        cmd = txt
        pose = None

        try:
            parsed = json.loads(txt)
            raw = parsed
            if isinstance(parsed, str):
                cmd = parsed
            elif isinstance(parsed, dict):
                cmd = str(parsed.get("command", parsed.get("cmd", parsed.get("action", ""))) or "").strip()
                if cmd == "":
                    if parsed.get("capture", False):
                        cmd = "capture"
                    elif parsed.get("quit", False):
                        cmd = "quit"
                    else:
                        cmd = txt
                pose = self._extract_pose6_from_obj(parsed)
            else:
                cmd = txt
        except Exception:
            pass

        return {"command": cmd, "tcp_pose_6dof": pose, "raw": raw}

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

    def request_tcp_pose(self) -> Optional[List[float]]:
        """
        Ask server for current TCP pose.
        Supported if Zeus server handles: {"action":"get_tcp_pose"}.
        """
        try:
            req = {"action": "get_tcp_pose"}
            self.sock.sendall(json.dumps(req).encode("utf-8"))
            txt = self._recv_text()
            try:
                obj = json.loads(txt)
            except Exception:
                obj = txt
            pose = self._extract_pose6_from_obj(obj)
            return pose
        except Exception:
            return None

    def send_pose_and_wait(self, joints: List[float], settle_time: float = 1.5) -> bool:
        """
        Backward-compatible API.
        Wait for 'capture' command, send joints, wait for settle.
        Returns True on success, False on quit/error.
        """
        ok, _, _ = self.send_pose_and_wait_with_tcp(
            joints=joints,
            settle_time=settle_time,
            query_tcp_if_missing=False,
        )
        return ok

    def send_pose_and_wait_with_tcp(
        self,
        joints: List[float],
        settle_time: float = 1.5,
        query_tcp_if_missing: bool = True,
    ) -> Tuple[bool, Optional[List[float]], str]:
        """
        Robot-controlled cycle with optional TCP extraction.

        Returns:
          (ok, tcp_pose_6dof, pose_source)
            pose_source in {"command", "query", "none", "quit", "error", "unknown"}
        """
        try:
            pkt = self.wait_for_command_packet()
            cmd = str(pkt.get("command", "")).strip().lower()

            if cmd == "quit":
                return False, None, "quit"

            if cmd != "capture":
                print(f"[RobotClient] Unknown command: {pkt.get('command')}")
                return False, None, "unknown"

            self.send_joint_command(joints)
            time.sleep(settle_time)

            tcp_pose = pkt.get("tcp_pose_6dof")
            if tcp_pose is not None:
                return True, tcp_pose, "command"

            if query_tcp_if_missing:
                tcp_pose = self.request_tcp_pose()
                if tcp_pose is not None:
                    return True, tcp_pose, "query"

            return True, None, "none"
        except Exception as e:
            print(f"[RobotClient] Error: {e}")
            return False, None, "error"
