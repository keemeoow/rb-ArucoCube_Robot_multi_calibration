# charuco_utils.py
"""
ChArUco 보드 검출 및 포즈 추정 유틸리티.
Eye-in-hand (그리퍼 카메라) 캘리브레이션에 사용.

보드 사양: 7x11, 체커 22mm, 마커 16mm, DICT_4X4_250
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from config import CharucoBoardConfig


class CharucoTarget:
    """ChArUco board detector and pose estimator."""

    def __init__(self, cfg: Optional[CharucoBoardConfig] = None):
        self.cfg = cfg or CharucoBoardConfig()

        # Create ArUco dictionary
        dict_name = self.cfg.dictionary_name
        dict_id = getattr(cv2.aruco, dict_name, None)
        if dict_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

        # OpenCV 4.7+ API
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(dict_id)

        self.board_ids = self._make_board_ids()
        self.board = self._create_charuco_board()
        self.board_id_set = set(int(x) for x in self.board_ids.tolist())

        # Detector parameters
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.det_params = cv2.aruco.DetectorParameters()
        else:
            self.det_params = cv2.aruco.DetectorParameters_create()

        # Detector (OpenCV 4.7+)
        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.det_params)

    def _make_board_ids(self) -> np.ndarray:
        """Build sequential marker IDs for this board."""
        start_id = int(self.cfg.marker_id_start)
        if start_id < 0:
            raise ValueError(f"marker_id_start must be >= 0, got {start_id}")

        num_markers = (int(self.cfg.squares_x) * int(self.cfg.squares_y)) // 2
        if num_markers <= 0:
            raise ValueError("Invalid board size; number of markers must be > 0")

        board_ids = np.arange(start_id, start_id + num_markers, dtype=np.int32)
        dict_size = int(self.dictionary.bytesList.shape[0])
        if int(board_ids[-1]) >= dict_size:
            raise ValueError(
                f"ChArUco marker IDs [{board_ids[0]}..{board_ids[-1]}] exceed "
                f"dictionary capacity ({dict_size} markers)"
            )
        return board_ids

    def _create_charuco_board(self):
        """Create ChArUco board with explicit marker IDs when supported."""
        if hasattr(cv2.aruco, "CharucoBoard"):
            try:
                return cv2.aruco.CharucoBoard(
                    (self.cfg.squares_x, self.cfg.squares_y),
                    self.cfg.square_length_m,
                    self.cfg.marker_length_m,
                    self.dictionary,
                    self.board_ids,
                )
            except TypeError:
                board = cv2.aruco.CharucoBoard(
                    (self.cfg.squares_x, self.cfg.squares_y),
                    self.cfg.square_length_m,
                    self.cfg.marker_length_m,
                    self.dictionary,
                )
        else:
            try:
                board = cv2.aruco.CharucoBoard_create(
                    self.cfg.squares_x,
                    self.cfg.squares_y,
                    self.cfg.square_length_m,
                    self.cfg.marker_length_m,
                    self.dictionary,
                    self.board_ids,
                )
            except TypeError:
                board = cv2.aruco.CharucoBoard_create(
                    self.cfg.squares_x,
                    self.cfg.squares_y,
                    self.cfg.square_length_m,
                    self.cfg.marker_length_m,
                    self.dictionary,
                )

        if hasattr(board, "setIds"):
            board.setIds(self.board_ids)
            return board

        if int(self.cfg.marker_id_start) != 0:
            raise RuntimeError(
                "This OpenCV build does not support custom ChArUco marker IDs. "
                "Please use OpenCV with CharucoBoard ids support."
            )
        return board

    def _filter_board_markers(self, marker_corners, marker_ids):
        """Keep only ArUco markers that belong to this ChArUco board."""
        if marker_ids is None or len(marker_ids) == 0:
            return None, None

        ids_flat = np.asarray(marker_ids).reshape(-1)
        keep_idx = [i for i, mid in enumerate(ids_flat) if int(mid) in self.board_id_set]
        if not keep_idx:
            return None, None

        kept_corners = [marker_corners[i] for i in keep_idx]
        kept_ids = np.array([int(ids_flat[i]) for i in keep_idx], dtype=np.int32).reshape(-1, 1)
        return kept_corners, kept_ids

    def detect(self, bgr: np.ndarray):
        """
        Detect ChArUco corners in image.

        Returns:
            charuco_corners: (N, 1, 2) array of corner positions, or None
            charuco_ids: (N, 1) array of corner IDs, or None
            n_corners: number of detected corners
            marker_corners: board-filtered ArUco marker corners
            marker_ids: board-filtered ArUco marker IDs
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers first
        if self.detector is not None:
            marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.det_params
            )

        marker_corners, marker_ids = self._filter_board_markers(marker_corners, marker_ids)
        if marker_ids is None or len(marker_ids) == 0:
            return None, None, 0, None, None

        # Interpolate ChArUco corners
        if hasattr(cv2.aruco, "interpolateCornersCharuco"):
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
        else:
            # OpenCV 4.7+
            ret, charuco_corners, charuco_ids = self.board.detectBoard(
                gray, markerCorners=marker_corners, markerIds=marker_ids
            )

        if ret < 4:
            return None, None, 0, marker_corners, marker_ids

        return charuco_corners, charuco_ids, int(ret), marker_corners, marker_ids

    def estimate_pose(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], int, float]:
        """
        Detect ChArUco board and estimate its pose.

        Returns:
            ok: True if pose estimation succeeded
            rvec: rotation vector (3,1)
            tvec: translation vector (3,1)
            n_corners: number of corners used
            reproj_err: reprojection error (px)
        """
        charuco_corners, charuco_ids, n_corners, _, _ = self.detect(bgr)

        if charuco_corners is None or n_corners < 4:
            return False, None, None, 0, float("inf")

        # Estimate pose
        if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board, K, D, None, None
            )
        else:
            # OpenCV 4.7+: use solvePnP with board object points
            obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is None or len(obj_pts) < 4:
                return False, None, None, 0, float("inf")
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)

        if not ok:
            return False, None, None, n_corners, float("inf")

        # Compute reprojection error
        obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is not None and len(obj_pts) > 0:
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1))
        else:
            err = float("inf")

        return True, rvec, tvec, n_corners, float(err)

    def draw_detected(self, bgr: np.ndarray, charuco_corners, charuco_ids):
        """Draw detected ChArUco corners on image."""
        out = bgr.copy()
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids)
        return out

    def draw_axis(self, bgr: np.ndarray, K, D, rvec, tvec, length=0.05):
        """Draw coordinate axis on image."""
        out = bgr.copy()
        cv2.drawFrameAxes(out, K, D, rvec, tvec, length)
        return out

    def generate_board_image(self, px_per_square: int = 40) -> np.ndarray:
        """Generate printable board image."""
        w = self.cfg.squares_x * px_per_square
        h = self.cfg.squares_y * px_per_square
        if hasattr(self.board, "generateImage"):
            return self.board.generateImage((w, h))
        else:
            return self.board.draw((w, h))
