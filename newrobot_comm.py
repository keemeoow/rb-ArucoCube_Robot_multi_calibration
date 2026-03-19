# newrobot_comm.py
"""
Robot communication helpers for Zeus calibration capture.

Adds a cycle protocol for:
  내려놓기(place) -> 위로 상승(capture pose) -> 카메라 촬영

The client supports both:
  - New protocol: {"command": "capture_cycle"}
  - Legacy protocol: {"command": "capture"}
"""

import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Pose6 = List[float]


def euler_deg_to_matrix(x_mm, y_mm, z_mm, rz_deg, ry_deg, rx_deg) -> np.ndarray:
    """
    Convert robot pose (x,y,z in mm, rz,ry,rx in deg) to 4x4 homogeneous matrix.
    Convention: ZYX extrinsic (Rz @ Ry @ Rx), translation in meters.
    """
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    Ry = np.array([
        [np.cos(ry), 0.0, np.sin(ry)],
        [0.0, 1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ], dtype=np.float64)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx), np.cos(rx)],
    ], dtype=np.float64)
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class RobotCycleClient:
    """Socket client for the cycle capture protocol."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.sock: Optional[socket.socket] = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"[RobotCycleClient] Connected to {self.host}:{self.port}")

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    @staticmethod
    def _to_float_pose6(vals: Any) -> Optional[Pose6]:
        try:
            arr = [float(x) for x in vals]
        except Exception:
            return None
        if len(arr) != 6:
            return None
        return arr

    @classmethod
    def _extract_pose6_from_obj(cls, obj: Any) -> Optional[Pose6]:
        if obj is None:
            return None

        if isinstance(obj, list):
            return cls._to_float_pose6(obj)

        if not isinstance(obj, dict):
            return None

        for k in [
            "tcp_pose_6dof",
            "robot_pose_6dof",
            "capture_pose_6dof",
            "pose_6dof",
            "pose",
            "tcp_pose",
        ]:
            if k in obj:
                p = cls._extract_pose6_from_obj(obj[k])
                if p is not None:
                    return p

        xyz_rzryrx = ["x", "y", "z", "rz", "ry", "rx"]
        if all(k in obj for k in xyz_rzryrx):
            try:
                return [
                    float(obj["x"]),
                    float(obj["y"]),
                    float(obj["z"]),
                    float(obj["rz"]),
                    float(obj["ry"]),
                    float(obj["rx"]),
                ]
            except Exception:
                pass

        xyz_rpy = ["x", "y", "z", "roll", "pitch", "yaw"]
        if all(k in obj for k in xyz_rpy):
            try:
                return [
                    float(obj["x"]),
                    float(obj["y"]),
                    float(obj["z"]),
                    float(obj["yaw"]),
                    float(obj["pitch"]),
                    float(obj["roll"]),
                ]
            except Exception:
                pass

        for k in ["payload", "data", "robot", "state", "result"]:
            if k in obj:
                p = cls._extract_pose6_from_obj(obj[k])
                if p is not None:
                    return p
        return None

    def _send_obj(self, obj: Dict[str, Any]):
        if self.sock is None:
            raise RuntimeError("Socket not connected")
        self.sock.sendall(json.dumps(obj).encode("utf-8"))

    def _recv_text(self) -> str:
        if self.sock is None:
            raise RuntimeError("Socket not connected")
        data = self.sock.recv(4096)
        if not data:
            raise RuntimeError("socket closed by peer")
        return data.decode("utf-8").strip()

    def wait_for_command_packet(self) -> Dict[str, Any]:
        """
        Normalize message into:
          {
            "command": <str>,
            "event": <optional str>,
            "tcp_pose_6dof": <optional pose6>,
            "raw": <decoded object or text>
          }
        """
        txt = self._recv_text()
        raw: Any = txt
        command = txt
        event = None
        pose = None

        try:
            parsed = json.loads(txt)
            raw = parsed
            if isinstance(parsed, str):
                command = parsed.strip()
            elif isinstance(parsed, dict):
                command = str(parsed.get("command", parsed.get("cmd", "")) or "").strip()
                event = str(parsed.get("event", "") or "").strip() or None

                if command == "":
                    if parsed.get("capture", False):
                        command = "capture"
                    elif parsed.get("quit", False):
                        command = "quit"
                    elif event:
                        command = event
                    else:
                        command = txt

                pose = self._extract_pose6_from_obj(parsed)
            else:
                command = txt
        except Exception:
            pass

        return {
            "command": command,
            "event": event,
            "tcp_pose_6dof": pose,
            "raw": raw,
        }

    def send_stop(self):
        self._send_obj({"action": "stop"})

    def send_cycle_command(
        self,
        place_pose_6dof: Pose6,
        capture_pose_6dof: Pose6,
        cycle_index: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        payload: Dict[str, Any] = {
            "action": "capture_cycle",
            "place_pose_6dof": [float(x) for x in place_pose_6dof],
            "capture_pose_6dof": [float(x) for x in capture_pose_6dof],
        }
        if cycle_index is not None:
            payload["cycle_index"] = int(cycle_index)
        if extra:
            payload["extra"] = extra
        self._send_obj(payload)

    def send_legacy_capture_command(self, pose_6dof: Pose6):
        d1, d2, d3, d4, d5, d6 = [float(x) for x in pose_6dof]
        self._send_obj({
            "status": "success",
            "action": "capture",
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d4": d4,
            "d5": d5,
            "d6": d6,
        })

    def request_tcp_pose(self) -> Optional[Pose6]:
        try:
            self._send_obj({"action": "get_tcp_pose"})
            pkt = self.wait_for_command_packet()
            return self._extract_pose6_from_obj(pkt.get("raw"))
        except Exception:
            return None

    def send_cycle_and_wait(
        self,
        place_pose_6dof: Pose6,
        capture_pose_6dof: Pose6,
        cycle_index: Optional[int] = None,
        post_done_wait: float = 0.0,
        query_tcp_if_missing: bool = True,
        done_timeout: float = 15.0,
    ) -> Tuple[bool, Optional[Pose6], str, Dict[str, Any]]:
        """
        Execute one robot cycle command after server's request.

        Returns:
          (ok, tcp_pose_6dof, pose_source, done_packet)
        """
        try:
            req = self.wait_for_command_packet()
            cmd = str(req.get("command", "")).strip().lower()

            if cmd == "quit":
                return False, None, "quit", req

            if cmd not in ["capture_cycle", "capture"]:
                print(f"[RobotCycleClient] Unknown server command: {req.get('command')}")
                return False, None, "unknown", req

            if cmd == "capture_cycle":
                self.send_cycle_command(
                    place_pose_6dof=place_pose_6dof,
                    capture_pose_6dof=capture_pose_6dof,
                    cycle_index=cycle_index,
                )
            else:
                # Legacy server compatibility
                self.send_legacy_capture_command(capture_pose_6dof)

            old_timeout = self.sock.gettimeout() if self.sock else None
            if self.sock is not None:
                self.sock.settimeout(float(done_timeout))

            done_pkt: Dict[str, Any]
            try:
                done_pkt = self.wait_for_command_packet()
            finally:
                if self.sock is not None:
                    self.sock.settimeout(old_timeout)

            done_cmd = str(done_pkt.get("command", "")).strip().lower()
            done_event = str(done_pkt.get("event", "") or "").strip().lower()

            if done_cmd == "quit" or done_event == "quit":
                return False, None, "quit", done_pkt

            tcp_pose = done_pkt.get("tcp_pose_6dof")
            if tcp_pose is not None:
                if post_done_wait > 0:
                    time.sleep(float(post_done_wait))
                return True, tcp_pose, "done_event", done_pkt

            if query_tcp_if_missing:
                tcp_pose = self.request_tcp_pose()
                if tcp_pose is not None:
                    if post_done_wait > 0:
                        time.sleep(float(post_done_wait))
                    return True, tcp_pose, "query", done_pkt

            if post_done_wait > 0:
                time.sleep(float(post_done_wait))
            return True, None, "none", done_pkt

        except Exception as e:
            print(f"[RobotCycleClient] Error: {e}")
            return False, None, "error", {}


class RobotClient(RobotCycleClient):
    """
    Backward-compatible alias.

    Existing code can still call send_pose_and_wait_with_tcp(),
    which maps to a single capture pose cycle.
    """

    def send_pose_and_wait_with_tcp(
        self,
        joints: List[float],
        settle_time: float = 1.5,
        query_tcp_if_missing: bool = True,
    ) -> Tuple[bool, Optional[Pose6], str]:
        ok, pose, source, _ = self.send_cycle_and_wait(
            place_pose_6dof=[float(x) for x in joints],
            capture_pose_6dof=[float(x) for x in joints],
            cycle_index=None,
            post_done_wait=settle_time,
            query_tcp_if_missing=query_tcp_if_missing,
        )
        return ok, pose, source
