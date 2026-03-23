# charuco_utils.py
"""
ChArUco board detection and pose estimation utilities.
Used for eye-in-hand (gripper camera) calibration.

Board spec: 7x11, square=22mm, marker=16mm, DICT_4X4_250
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

        # Create CharucoBoard
        if hasattr(cv2.aruco, "CharucoBoard"):
            # OpenCV 4.7+
            self.board = cv2.aruco.CharucoBoard(
                (self.cfg.squares_x, self.cfg.squares_y),
                self.cfg.square_length_m,
                self.cfg.marker_length_m,
                self.dictionary,
            )
        else:
            # Older OpenCV
            self.board = cv2.aruco.CharucoBoard_create(
                self.cfg.squares_x,
                self.cfg.squares_y,
                self.cfg.square_length_m,
                self.cfg.marker_length_m,
                self.dictionary,
            )

        # Detector parameters
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.det_params = cv2.aruco.DetectorParameters()
        else:
            self.det_params = cv2.aruco.DetectorParameters_create()

        # Detector (OpenCV 4.7+)
        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.det_params)

    def detect(self, bgr: np.ndarray):
        """
        Detect ChArUco corners in image.

        Returns:
            charuco_corners: (N, 1, 2) array of corner positions, or None
            charuco_ids: (N, 1) array of corner IDs, or None
            n_corners: number of detected corners
            marker_corners: raw ArUco marker corners
            marker_ids: raw ArUco marker IDs
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers first
        if self.detector is not None:
            marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.det_params
            )

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
