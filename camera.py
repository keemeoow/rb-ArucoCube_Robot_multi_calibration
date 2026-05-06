# camera.py
"""
RealSense camera wrapper with threaded capture.
Supports multiple cameras simultaneously.

Each camera keeps a short ring buffer of (timestamp, color, depth) frames
so callers can retrieve the frame closest to a target timestamp via
`get_at()`, enabling software synchronization across multiple cameras.

`start()` is resilient: on "Frame didn't arrive" timeout it performs a
hardware reset of the device (by serial), waits for USB re-enumeration,
rebuilds the pipeline, and retries up to 3 times. Warmup uses a longer
10-second timeout to tolerate slow first-frame delivery on multi-camera
USB hubs.
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple, Dict

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """Thread-safe RealSense camera capture with a short frame ring buffer."""

    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        use_color: bool = True,
        use_depth: bool = False,
        align_depth_to_color: bool = True,
        warmup_frames: int = 10,
        buffer_size: int = 8,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_color = bool(use_color)
        self.use_depth = bool(use_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.warmup_frames = int(warmup_frames)
        self.buffer_size = max(1, int(buffer_size))

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buf: deque = deque(maxlen=self.buffer_size)

        self._build_pipeline()

    def _build_pipeline(self):
        """(Re)create pipeline/config/align — needed after hardware_reset()."""
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial)

        if self.use_color:
            self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.use_depth:
            self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.align = (
            rs.align(rs.stream.color)
            if (self.use_depth and self.align_depth_to_color and self.use_color)
            else None
        )

    def _hardware_reset(self, wait_s: float = 6.0):
        """Hardware-reset device by serial and wait for USB re-enumeration."""
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial:
                    dev.hardware_reset()
                    break
        except Exception:
            pass

        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                serials = [
                    d.get_info(rs.camera_info.serial_number)
                    for d in rs.context().query_devices()
                ]
                if self.serial in serials:
                    time.sleep(0.8)  # extra settle time after re-enumeration
                    return
            except Exception:
                continue

    @staticmethod
    def list_devices() -> Dict[str, str]:
        """Return {serial: name} for all connected RealSense devices."""
        ctx = rs.context()
        out = {}
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            out[serial] = name
        return out

    def start(self, max_attempts: int = 3, warmup_timeout_ms: int = 10000):
        last_err = None
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    print(f"[WARN] cam {self.serial}: retrying after hardware reset "
                          f"(attempt {attempt + 1}/{max_attempts})")
                    self._hardware_reset()
                    self._build_pipeline()
                self.pipeline.start(self.cfg)
                for _ in range(self.warmup_frames):
                    self.pipeline.wait_for_frames(timeout_ms=warmup_timeout_ms)
                self._running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
                return
            except RuntimeError as e:
                last_err = e
                print(f"[WARN] cam {self.serial}: start failed "
                      f"(attempt {attempt + 1}/{max_attempts}): {e}")
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
        raise RuntimeError(
            f"Camera {self.serial} failed to start after {max_attempts} attempts "
            f"(last error: {last_err}). Try unplugging/replugging USB or check "
            f"`rs-enumerate-devices` output."
        )

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                if self.align is not None:
                    frames = self.align.process(frames)

                color = frames.get_color_frame() if self.use_color else None
                depth = frames.get_depth_frame() if self.use_depth else None

                if self.use_color and color is None:
                    continue

                ts_ms = None
                if color is not None:
                    ts_ms = float(color.get_timestamp())
                elif depth is not None:
                    ts_ms = float(depth.get_timestamp())

                color_arr = None if color is None else np.asanyarray(color.get_data()).copy()
                depth_arr = None if depth is None else np.asanyarray(depth.get_data()).copy()

                with self._lock:
                    self._buf.append((ts_ms, color_arr, depth_arr))
            except Exception:
                time.sleep(0.005)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Return (color_bgr, depth_u16, timestamp_ms) - copies of most recent frame."""
        with self._lock:
            if not self._buf:
                return None, None, None
            ts, c, d = self._buf[-1]
        return (None if c is None else c.copy()), (None if d is None else d.copy()), ts

    def get_at(self, target_ts_ms: float
               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Return the buffered frame whose timestamp is closest to `target_ts_ms`.

        Used for software-synchronized multi-camera capture: pick a reference
        timestamp across cameras and have each camera return its closest frame.
        """
        with self._lock:
            if not self._buf:
                return None, None, None
            best = min(
                self._buf,
                key=lambda x: abs(x[0] - target_ts_ms) if x[0] is not None else float("inf"),
            )
            ts, c, d = best
        return (None if c is None else c.copy()), (None if d is None else d.copy()), ts
