from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def filter_marker_ids(corners_list, ids, allowed_ids: Sequence[int]):
    if ids is None or len(ids) == 0:
        return [], None
    allowed = set(int(x) for x in allowed_ids)
    filt_corners = []
    filt_ids = []
    ids_flat = np.asarray(ids).reshape(-1)
    for corners, mid in zip(corners_list, ids_flat):
        if int(mid) in allowed:
            filt_corners.append(corners)
            filt_ids.append(int(mid))
    if not filt_ids:
        return [], None
    return filt_corners, np.asarray(filt_ids, dtype=np.int32)


def _expand_quad(corners, pad_px: float) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return np.round(pts).astype(np.int32)
    if float(pad_px) <= 0:
        return np.round(pts).astype(np.int32)
    center = np.mean(pts, axis=0)
    radii = np.linalg.norm(pts - center[None, :], axis=1)
    mean_radius = float(np.mean(radii)) if radii.size > 0 else 1.0
    scale = 1.0 + float(pad_px) / max(mean_radius, 1.0)
    expanded = center[None, :] + (pts - center[None, :]) * scale
    return np.round(expanded).astype(np.int32)


def mask_board_marker_regions(bgr: np.ndarray,
                              board_marker_corners,
                              pad_px: float = 6.0,
                              fill_value: int = 127) -> np.ndarray:
    if bgr is None:
        return bgr
    if board_marker_corners is None or len(board_marker_corners) == 0:
        return bgr.copy()
    masked = bgr.copy()
    fill_color = (int(fill_value), int(fill_value), int(fill_value))
    for corners in board_marker_corners:
        quad = _expand_quad(corners, pad_px)
        if quad.shape[0] >= 3:
            cv2.fillConvexPoly(masked, quad, fill_color, lineType=cv2.LINE_AA)
    return masked


def detect_cube_markers_in_frame(bgr: np.ndarray,
                                 cube,
                                 cube_ids: Sequence[int],
                                 charuco=None,
                                 is_gripper: bool = False,
                                 board_mask_pad_px: float = 6.0) -> Dict[str, object]:
    board_mkr_corners = None
    board_mkr_ids = None
    ch_corners = None
    ch_ids = None
    charuco_detect_n = 0
    board_mask_applied = False

    cube_img = bgr
    if is_gripper and charuco is not None:
        try:
            ch_corners, ch_ids, charuco_detect_n, board_mkr_corners, board_mkr_ids = charuco.detect(bgr)
        except Exception:
            ch_corners, ch_ids, charuco_detect_n, board_mkr_corners, board_mkr_ids = None, None, 0, None, None
        if board_mkr_corners is not None and len(board_mkr_corners) > 0:
            cube_img = mask_board_marker_regions(bgr, board_mkr_corners, pad_px=board_mask_pad_px)
            board_mask_applied = True

    raw_corners, raw_ids = cube.detect(cube_img)
    raw_ids_list = [] if raw_ids is None else [int(x) for x in np.asarray(raw_ids).reshape(-1)]
    corners, ids = filter_marker_ids(raw_corners, raw_ids, cube_ids)
    filtered_ids_list = [] if ids is None else [int(x) for x in np.asarray(ids).reshape(-1)]

    return {
        "cube_image": cube_img,
        "corners": corners,
        "ids": ids,
        "raw_ids": raw_ids_list,
        "filtered_ids": filtered_ids_list,
        "board_mkr_corners": board_mkr_corners,
        "board_mkr_ids": board_mkr_ids,
        "ch_corners": ch_corners,
        "ch_ids": ch_ids,
        "charuco_detect_n": int(charuco_detect_n),
        "board_mask_applied": bool(board_mask_applied),
    }
