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

        dict_name = self.cfg.dictionary_name
        dict_id = getattr(cv2.aruco, dict_name, None)
        if dict_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(dict_id)

        self.board_ids = self._make_board_ids()  # physical marker ids on the printed board
        self.custom_ids_supported = False
        self.board = self._create_charuco_board()
        self.board_id_set = set(int(x) for x in self.board_ids.tolist())
        self.default_board_id_set = set(range(len(self.board_ids)))
        self._using_id_remap = (not self.custom_ids_supported) and int(self.cfg.marker_id_start) != 0

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.det_params = cv2.aruco.DetectorParameters()
        else:
            self.det_params = cv2.aruco.DetectorParameters_create()

        # ChArUco docs recommend disabling marker corner refinement for ChArUco interpolation.
        if hasattr(cv2.aruco, "CORNER_REFINE_NONE") and hasattr(self.det_params, "cornerRefinementMethod"):
            self.det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.det_params)

        self.charuco_detector = None
        if hasattr(cv2.aruco, "CharucoDetector"):
            try:
                if hasattr(cv2.aruco, "CharucoParameters"):
                    charuco_params = cv2.aruco.CharucoParameters()
                    self.charuco_detector = cv2.aruco.CharucoDetector(
                        self.board, charuco_params, self.det_params
                    )
                else:
                    self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
            except Exception:
                self.charuco_detector = None

    def _make_board_ids(self) -> np.ndarray:
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

    def _create_default_board(self):
        if hasattr(cv2.aruco, "CharucoBoard"):
            return cv2.aruco.CharucoBoard(
                (self.cfg.squares_x, self.cfg.squares_y),
                self.cfg.square_length_m,
                self.cfg.marker_length_m,
                self.dictionary,
            )
        return cv2.aruco.CharucoBoard_create(
            self.cfg.squares_x,
            self.cfg.squares_y,
            self.cfg.square_length_m,
            self.cfg.marker_length_m,
            self.dictionary,
        )

    def _create_charuco_board(self):
        """
        Try to create a board with explicit marker IDs.
        If the current OpenCV Python binding cannot store custom IDs, fall back to a
        default-ID board and later remap detected marker IDs into the board-local range.
        """
        if hasattr(cv2.aruco, "CharucoBoard"):
            try:
                board = cv2.aruco.CharucoBoard(
                    (self.cfg.squares_x, self.cfg.squares_y),
                    self.cfg.square_length_m,
                    self.cfg.marker_length_m,
                    self.dictionary,
                    self.board_ids,
                )
                self.custom_ids_supported = True
                return board
            except TypeError:
                board = self._create_default_board()
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
                self.custom_ids_supported = True
                return board
            except TypeError:
                board = self._create_default_board()

        if hasattr(board, "setIds"):
            try:
                board.setIds(self.board_ids)
                self.custom_ids_supported = True
                return board
            except Exception:
                pass

        self.custom_ids_supported = False
        return board

    def _filter_board_markers(self, marker_corners, marker_ids):
        if marker_ids is None or len(marker_ids) == 0:
            return None, None, None

        ids_flat = np.asarray(marker_ids).reshape(-1)
        keep_idx = [i for i, mid in enumerate(ids_flat) if int(mid) in self.board_id_set]
        if not keep_idx:
            return None, None, None

        kept_corners = [marker_corners[i] for i in keep_idx]
        kept_ids_phys = np.array([int(ids_flat[i]) for i in keep_idx], dtype=np.int32).reshape(-1, 1)

        if self._using_id_remap:
            start_id = int(self.cfg.marker_id_start)
            kept_ids_for_board = (kept_ids_phys.astype(np.int32) - start_id).reshape(-1, 1)
        else:
            kept_ids_for_board = kept_ids_phys.copy()

        return kept_corners, kept_ids_phys, kept_ids_for_board

    def detect(self, bgr: np.ndarray):
        """
        Returns:
            charuco_corners, charuco_ids, n_corners, marker_corners, marker_ids(physical ids)
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.det_params
            )

        marker_corners, marker_ids_phys, marker_ids_for_board = self._filter_board_markers(
            marker_corners, marker_ids
        )
        if marker_ids_phys is None or len(marker_ids_phys) == 0:
            return None, None, 0, None, None

        charuco_corners, charuco_ids = None, None
        n_corners = 0

        # Preferred new API (OpenCV 4.7+): detectBoard takes image only
        if self.charuco_detector is not None and not self._using_id_remap:
            try:
                charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(gray)
                n_corners = 0 if charuco_ids is None else len(charuco_ids)
            except Exception:
                charuco_corners, charuco_ids, n_corners = None, None, 0

        # Compatible fallback, also works with ID remapping
        if charuco_ids is None and hasattr(cv2.aruco, "interpolateCornersCharuco"):
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids_for_board,
                gray,
                self.board,
            )
            n_corners = int(ret) if ret is not None else 0

        if charuco_ids is None or n_corners < 4:
            return None, None, 0, marker_corners, marker_ids_phys

        return charuco_corners, charuco_ids, int(n_corners), marker_corners, marker_ids_phys

    def estimate_pose(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], int, float]:
        charuco_corners, charuco_ids, n_corners, _, _ = self.detect(bgr)
        if charuco_corners is None or n_corners < 4:
            return False, None, None, 0, float("inf")

        ok, rvec, tvec = False, None, None

        if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
            try:
                rvec0 = np.zeros((3, 1), dtype=np.float64)
                tvec0 = np.zeros((3, 1), dtype=np.float64)
                out = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, K, D, rvec0, tvec0
                )
                # Python bindings vary: sometimes bool only, sometimes (bool, rvec, tvec)
                if isinstance(out, tuple):
                    if len(out) == 3:
                        ok, rvec, tvec = out
                    elif len(out) == 1:
                        ok = bool(out[0])
                        rvec, tvec = rvec0, tvec0
                else:
                    ok = bool(out)
                    rvec, tvec = rvec0, tvec0
            except Exception:
                ok = False

        if not ok:
            obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is None or len(obj_pts) < 4:
                return False, None, None, 0, float("inf")
            # ChArUco corners are coplanar; SOLVEPNP_IPPE handles the planar case
            # with >=4 points. The default ITERATIVE flag routes <6 points through
            # DLT, which OpenCV >=4.11 rejects ("needs at least 6 points").
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                return False, None, None, n_corners, float("inf")
        else:
            obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)

        if obj_pts is not None and len(obj_pts) > 0:
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1))
        else:
            err = float("inf")

        return True, rvec, tvec, n_corners, float(err)

    def draw_detected(self, bgr: np.ndarray, charuco_corners, charuco_ids):
        out = bgr.copy()
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids)
        return out

    def draw_axis(self, bgr: np.ndarray, K, D, rvec, tvec, length=0.05):
        out = bgr.copy()
        cv2.drawFrameAxes(out, K, D, rvec, tvec, length)
        return out

    def generate_board_image(self, px_per_square: int = 40) -> np.ndarray:
        if self._using_id_remap:
            raise RuntimeError(
                "This OpenCV build cannot render a custom-ID ChArUco board from Python. "
                "Detection works via ID remapping, but printable board generation here would be wrong."
            )
        w = self.cfg.squares_x * px_per_square
        h = self.cfg.squares_y * px_per_square
        if hasattr(self.board, "generateImage"):
            return self.board.generateImage((w, h))
        return self.board.draw((w, h))
