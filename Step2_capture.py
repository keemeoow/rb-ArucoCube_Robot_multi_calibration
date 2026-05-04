# Step2_capture.py
"""
Step 2: 멀티카메라 캘리브레이션용 캡처 수집.

파이프라인:
  1. 로봇이 큐브를 놓고 `set`을 실행하면 set 기준 pose를 저장한다.
  2. 같은 set에서 그리퍼 카메라를 여러 자세로 이동시키며 촬영한다.
  3. 각 이벤트에서 모든 카메라(그리퍼 + 고정)가 동시에 color/depth를 저장한다.
  4. ArUco cube / gripper ChArUco를 즉시 검출하고 pose 후보와 품질 지표를 meta.json에 기록한다.
  5. `set_index`, robot pose, set_cube_center_6dof, capture gate 결과를 함께 저장한다.

실행 명령어:
python Step2_capture.py \
    --root_folder ./data/session \
    --intrinsics_dir ./intrinsics \
    --use_robot --manual_robot \
    --robot_ip 192.168.0.23 --robot_port 12348 \
    --settle_time 1.5 \
    --show \
    --min_markers 1 --min_cams_with_cube 1

[서버코드 작동법] (큐브 위치별 멀티 캘리브레이션)
# ── 1. 큐브를 바닥에 놓고 위치 저장 ──
> set                         # 큐브 위치 #0 저장 (TCP, 관절값, 큐브 중점)

# ── 2. 그리퍼 카메라 위치를 다양하게 이동하며 촬영 ──
> p z,300                     # 위로 이동 (그리퍼 카메라가 큐브를 볼 수 있게)
> c                           # 촬영 (set_index=0, 관절값 포함 PC 전송)
> p x,100
> p rz,45
> c                           # 같은 큐브 위치에서 다른 각도로 촬영

# ── 3. 큐브를 새 위치로 옮기고 반복 ──
> set                         # 큐브 위치 #1 저장
> p z,300
> c                           # set_index=1로 촬영
> ...

# ── 4. 종료 ──
> q

저장 파일:
  - meta.json               : 캡처별 상세 (robot pose, set_index, set_cube_center_6dof, cube/board quality)
  - capture_waypoints.json  : 웨이포인트 (set_joints/tcp, place_joints, capture_joints)

참고:
  - depth 저장은 기본 ON이다. 끄려면 `--no-save-depth`를 사용한다.
  - downstream Step3는 여기 저장된 set_cube_center_6dof와 depth 품질 지표를 prior/selection에 사용한다.

PC waypoint step mode:
  python Step2_capture.py \
      --root_folder ./data/session \
      --intrinsics_dir ./intrinsics \
      --use_robot \
      --waypoint_file ./data/session/capture_waypoints.json

  Controls in the preview window:
    ENTER : move robot to the next saved waypoint
    SPACE : capture at the current robot pose
    ESC/q : quit

"""

import os
import sys as _sys_top
import json
import time
import argparse
import select as _select
import threading as _threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera import RealSenseCamera
from aruco_cube import ArucoCubeTarget, depth_metrics_to_fields, rodrigues_to_Rt
from charuco_utils import CharucoTarget
from config import CubeConfig, CharucoBoardConfig, get_default_cube_config
from calibration_runtime_utils import resolve_cube_config_for_run
from capture_detection_utils import detect_cube_markers_in_frame
from cube_config_utils import (
    cube_config_mismatch_keys,
    cube_config_to_dict,
    cube_configs_equivalent,
    load_cube_config_from_meta,
)
from robot_comm import PlaceCaptureClient, euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _as_pose6(value) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != 6:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def normalize_waypoint_file_payload(payload) -> Tuple[List[dict], dict]:
    if isinstance(payload, list):
        return payload, {}
    if isinstance(payload, dict):
        waypoints = payload.get("waypoints")
        if isinstance(waypoints, list):
            return waypoints, payload
    raise ValueError("Unsupported waypoint file format")


def resolve_waypoint_motion(wp: dict, bundle: dict) -> Tuple[List[float], List[float], str, str]:
    # Legacy generator format: {"place": [...], "capture": [...]}
    place_pose = _as_pose6(wp.get("place"))
    capture_pose = _as_pose6(wp.get("capture"))
    if place_pose is not None and capture_pose is not None:
        return place_pose, capture_pose, "tcp", "tcp"

    # Replay format mirrored from robot server: prefer joints when available.
    place_pose = _as_pose6(wp.get("place_joints"))
    capture_pose = _as_pose6(wp.get("capture_joints"))
    if place_pose is not None and capture_pose is not None:
        return place_pose, capture_pose, "joint", "joint"

    # Fallbacks when only TCP is available.
    place_pose = _as_pose6(wp.get("place_tcp")) or _as_pose6(bundle.get("set_tcp"))
    capture_pose = _as_pose6(wp.get("capture_tcp"))
    if place_pose is not None and capture_pose is not None:
        return place_pose, capture_pose, "tcp", "tcp"

    raise ValueError(f"Waypoint is missing usable place/capture poses: keys={sorted(wp.keys())}")


def annotate_image(bgr, cube, cam_idx, is_gripper, n_markers, ids, corners,
                    board_mkr_corners=None, board_mkr_ids=None,
                    ch_corners=None, ch_ids=None):
    """마커 오버레이 및 정보 텍스트를 이미지에 그림."""
    out = bgr.copy()

    # Gripper camera: draw board ArUco markers (DICT_4X4_250)
    n_board = 0
    if is_gripper and board_mkr_corners is not None and board_mkr_ids is not None:
        n_board = len(board_mkr_ids)
        try:
            cv2.aruco.drawDetectedMarkers(out, board_mkr_corners, board_mkr_ids)
        except Exception:
            pass

    # Gripper camera: draw ChArUco interpolated corners
    n_charuco = 0
    if is_gripper and ch_corners is not None and ch_ids is not None:
        n_charuco = len(ch_ids)
        try:
            cv2.aruco.drawDetectedCornersCharuco(out, ch_corners, ch_ids)
        except Exception:
            pass

    # Draw cube markers
    if ids is not None and len(corners) > 0:
        try:
            draw_ids = ids.reshape(-1, 1) if getattr(ids, "ndim", 1) == 1 else ids
            cv2.aruco.drawDetectedMarkers(out, corners, draw_ids)
        except Exception:
            pass

    role = "GRIPPER" if is_gripper else "FIXED"
    ids_txt = ",".join(str(int(x)) for x in ids) if ids is not None and len(ids) > 0 else "-"
    board_txt = ""
    if is_gripper:
        board_txt = f" board={n_board}mkr ch={n_charuco}cor"
    lines = [
        f"cam{cam_idx} [{role}]",
        f"markers={n_markers} ids={ids_txt}{board_txt}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (4, y - 18), (10 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22
    return out


def wait_for_start_command_capture(cams, cam_order, gripper_cam_idx,
                                     extra_lines: Optional[List[str]] = None) -> bool:
    """캘리브레이션 캡처 시작 전 cv2 프리뷰 + 'start' 입력 대기.

    아르코 큐브 / charuco 오버레이는 표시하지 않고, 단순 4-캠 raw 프리뷰 만
    띄운다 (사용자가 카메라 시야/노출만 확인). 터미널에서:
      start  -> 캡처 메인 루프 진입 (창은 닫지 않고 후속 모드에서 같은 창 갱신)
      quit   -> 캡처 시작 안 하고 종료 (창 닫음)
    Returns: True 시작 / False 사용자 취소.
    """
    print("")
    print("=" * 60)
    print(" Live preview — type 'start' (then ENTER) in this terminal to begin")
    print(" or type 'quit' / press q in the preview window to abort")
    print("=" * 60)
    if extra_lines:
        for ln in extra_lines:
            print(" " + ln)

    start_event = _threading.Event()
    quit_event = _threading.Event()

    def _stdin_reader():
        while not (start_event.is_set() or quit_event.is_set()):
            try:
                r, _, _ = _select.select([_sys_top.stdin], [], [], 0.2)
                if not r:
                    continue
                line = _sys_top.stdin.readline()
                if not line:
                    quit_event.set(); return
                token = line.strip().lower()
                if token == "start":
                    start_event.set(); return
                if token in ("quit", "q", "exit"):
                    quit_event.set(); return
                if token:
                    print(f"  type 'start' or 'quit' (got: {token!r})")
            except Exception:
                quit_event.set(); return

    t = _threading.Thread(target=_stdin_reader, daemon=True)
    t.start()

    win = "Capture Preview"
    while not start_event.is_set() and not quit_event.is_set():
        tiles = []
        tile_h = tile_w = None
        for ci in cam_order:
            cam = cams.get(ci)
            color = None
            if cam is not None:
                color, _depth, _ts = cam.get_latest()
            if color is not None:
                if tile_h is None:
                    tile_h, tile_w = color.shape[:2]
                disp = color.copy()
                tag = "GRIP" if (gripper_cam_idx is not None and ci == gripper_cam_idx) else "FIX"
                col = (0, 200, 255) if tag == "GRIP" else (0, 255, 0)
                cv2.putText(disp, f"cam{ci} [{tag}]", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                tiles.append(disp)
            else:
                if tile_h is None:
                    tile_h, tile_w = 480, 640
                blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(blank, f"cam{ci} N/A", (20, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                tiles.append(blank)
        while len(tiles) < 4:
            tiles.append(np.zeros((tile_h or 480, tile_w or 640, 3), dtype=np.uint8))
        tiles = tiles[:4]
        top = cv2.hconcat([tiles[0], tiles[1]])
        bot = cv2.hconcat([tiles[2], tiles[3]])
        quad = cv2.vconcat([top, bot])

        # footer
        foot_h = 28 + 26 * (1 + (len(extra_lines) if extra_lines else 0))
        foot = np.zeros((foot_h, quad.shape[1], 3), dtype=np.uint8)
        cv2.putText(foot, "[WAITING] Type 'start' + ENTER in terminal to begin",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        if extra_lines:
            y = 48
            for ln in extra_lines:
                cv2.putText(foot, ln, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                y += 24
        quad = cv2.vconcat([quad, foot])
        h2 = int(quad.shape[0] * 0.6); w2 = int(quad.shape[1] * 0.6)
        cv2.imshow(win, cv2.resize(quad, (w2, h2)))

        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key == ord('q'):
            quit_event.set()
            break

    if start_event.is_set():
        print("[start] confirmed, proceeding...")
        # 후속 모드 진입 전까지 창이 "응답 없음"으로 빠지지 않도록 한 번 펌프.
        try:
            cv2.waitKey(1)
        except Exception:
            pass
        return True
    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    print("[abort] user cancelled before start.")
    return False


def make_quad_image(frames_dict, cam_order, cube, gripper_cam_idx):
    """4개 카메라로부터 마커 오버레이가 포함된 2x2 분할 이미지를 생성."""
    tiles = []
    tile_h, tile_w = None, None

    for ci in cam_order:
        fr = frames_dict.get(ci)
        if fr is not None and fr.get("color") is not None:
            img = fr["color"]
            if tile_h is None:
                tile_h, tile_w = img.shape[:2]
            annotated = annotate_image(
                img, cube, ci,
                is_gripper=(ci == gripper_cam_idx),
                n_markers=fr.get("n_markers", 0),
                ids=fr.get("ids_np"),
                corners=fr.get("corners", []),
                board_mkr_corners=fr.get("board_mkr_corners"),
                board_mkr_ids=fr.get("board_mkr_ids"),
                ch_corners=fr.get("ch_corners"),
                ch_ids=fr.get("ch_ids"),
            )
            tiles.append(annotated)
        else:
            if tile_h is None:
                tile_h, tile_w = 480, 640
            blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.putText(blank, f"cam{ci} N/A", (20, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            tiles.append(blank)

    while len(tiles) < 4:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    tiles = tiles[:4]

    top = cv2.hconcat([tiles[0], tiles[1]])
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    return cv2.vconcat([top, bottom])


def make_capture_gate_config(args) -> dict:
    return {
        "min_cams_with_cube": int(args.min_cams_with_cube),
        "min_fixed_cams_with_cube": int(args.min_fixed_cams_with_cube),
        "min_cube_pnp_ok_cams": int(args.min_cube_pnp_ok_cams),
        "min_fixed_cube_pnp_ok_cams": int(args.min_fixed_cube_pnp_ok_cams),
        "min_gripper_charuco_corners": int(args.min_gripper_charuco_corners),
        "require_gripper_cube_pnp": bool(args.require_gripper_cube_pnp),
        "require_gripper_depth_valid": bool(args.require_gripper_depth_valid),
        "max_gripper_depth_plane_mean_mm": float(args.max_gripper_depth_plane_mean_mm),
        "max_capture_span_ms": float(args.max_capture_span_ms),
    }


def evaluate_capture_gate(frames_dict: Dict[int, dict],
                          gate_cfg: dict,
                          gripper_cam_idx: Optional[int] = None) -> dict:
    cams_with_cube = 0
    fixed_visible = 0
    cube_pnp_ok_cams = 0
    fixed_cube_pnp_ok_cams = 0
    depth_valid_cams = 0
    fixed_depth_valid_cams = 0
    capture_ts = []
    per_camera = {}
    gripper_markers = 0
    gripper_charuco_corners = 0
    gripper_cube_pnp_ok = False
    gripper_depth_valid = False
    gripper_depth_plane_mean_mm = None

    for ci, fr in frames_dict.items():
        n_markers = int(fr.get("n_markers", 0))
        cube_visible = bool(fr.get("ok", False))
        cube_pnp = fr.get("cube_pnp")
        cube_pnp_ok = bool(cube_pnp is not None)
        depth_valid = bool(cube_pnp and cube_pnp.get("depth_valid"))
        depth_plane_mean_mm = None if not cube_pnp else cube_pnp.get("depth_plane_mean_mm")
        ts_ms = fr.get("ts_ms")
        if ts_ms is not None:
            capture_ts.append(float(ts_ms))
        if cube_visible:
            cams_with_cube += 1
            if gripper_cam_idx is None or int(ci) != int(gripper_cam_idx):
                fixed_visible += 1
        if cube_pnp_ok:
            cube_pnp_ok_cams += 1
            if gripper_cam_idx is None or int(ci) != int(gripper_cam_idx):
                fixed_cube_pnp_ok_cams += 1
        if depth_valid:
            depth_valid_cams += 1
            if gripper_cam_idx is None or int(ci) != int(gripper_cam_idx):
                fixed_depth_valid_cams += 1
        if gripper_cam_idx is not None and int(ci) == int(gripper_cam_idx):
            gripper_markers = int(n_markers)
            ch_ids = fr.get("ch_ids")
            gripper_charuco_corners = 0 if ch_ids is None else len(ch_ids)
            gripper_cube_pnp_ok = cube_pnp_ok
            gripper_depth_valid = depth_valid
            if depth_plane_mean_mm is not None:
                gripper_depth_plane_mean_mm = float(depth_plane_mean_mm)
        per_camera[int(ci)] = {
            "n_markers": n_markers,
            "cube_visible": cube_visible,
            "cube_pnp_ok": cube_pnp_ok,
            "depth_valid": depth_valid,
        }

    capture_span_ms = (max(capture_ts) - min(capture_ts)) if len(capture_ts) >= 2 else 0.0
    reasons = []
    min_cams_with_cube = int(gate_cfg.get("min_cams_with_cube", 0))
    min_fixed_cams_with_cube = int(gate_cfg.get("min_fixed_cams_with_cube", 0))
    min_cube_pnp_ok_cams = int(gate_cfg.get("min_cube_pnp_ok_cams", 0))
    min_fixed_cube_pnp_ok_cams = int(gate_cfg.get("min_fixed_cube_pnp_ok_cams", 0))
    min_gripper_charuco_corners = int(gate_cfg.get("min_gripper_charuco_corners", 0))
    require_gripper_cube_pnp = bool(gate_cfg.get("require_gripper_cube_pnp", False))
    require_gripper_depth_valid = bool(gate_cfg.get("require_gripper_depth_valid", False))
    max_gripper_depth_plane_mean_mm = float(gate_cfg.get("max_gripper_depth_plane_mean_mm", 0.0))
    max_capture_span_ms = float(gate_cfg.get("max_capture_span_ms", 0.0))

    if cams_with_cube < min_cams_with_cube:
        reasons.append(
            "cube-visible cams {} < required {}".format(cams_with_cube, min_cams_with_cube)
        )
    if fixed_visible < min_fixed_cams_with_cube:
        reasons.append(
            "fixed cube-visible cams {} < required {}".format(fixed_visible, min_fixed_cams_with_cube)
        )
    if cube_pnp_ok_cams < min_cube_pnp_ok_cams:
        reasons.append(
            "cube_pnp-ok cams {} < required {}".format(cube_pnp_ok_cams, min_cube_pnp_ok_cams)
        )
    if fixed_cube_pnp_ok_cams < min_fixed_cube_pnp_ok_cams:
        reasons.append(
            "fixed cube_pnp-ok cams {} < required {}".format(
                fixed_cube_pnp_ok_cams, min_fixed_cube_pnp_ok_cams
            )
        )
    if require_gripper_cube_pnp and not gripper_cube_pnp_ok:
        reasons.append("gripper cube_pnp missing")
    if min_gripper_charuco_corners > 0 and gripper_charuco_corners < min_gripper_charuco_corners:
        reasons.append(
            "gripper charuco corners {} < required {}".format(
                gripper_charuco_corners, min_gripper_charuco_corners
            )
        )
    if require_gripper_depth_valid and require_gripper_cube_pnp and gripper_cube_pnp_ok and not gripper_depth_valid:
        reasons.append("gripper depth support invalid")
    if (
        require_gripper_depth_valid
        and gripper_depth_valid
        and max_gripper_depth_plane_mean_mm > 0
        and gripper_depth_plane_mean_mm is not None
        and float(gripper_depth_plane_mean_mm) > max_gripper_depth_plane_mean_mm
    ):
        reasons.append(
            "gripper depth plane {:.1f}mm > {:.1f}mm".format(
                float(gripper_depth_plane_mean_mm), max_gripper_depth_plane_mean_mm
            )
        )
    if max_capture_span_ms > 0 and capture_span_ms > float(max_capture_span_ms):
        reasons.append(
            "timestamp span {:.1f}ms > {:.1f}ms".format(
                float(capture_span_ms), float(max_capture_span_ms)
            )
        )

    status = "PASS" if not reasons else "FAIL"
    reason = " | ".join(reasons) if reasons else "capture gate satisfied"
    return {
        "pass": bool(not reasons),
        "status": status,
        "reason": reason,
        "reasons": reasons,
        "cams_with_cube": int(cams_with_cube),
        "min_cams_with_cube": int(min_cams_with_cube),
        "capture_span_ms": float(capture_span_ms),
        "max_capture_span_ms": float(max_capture_span_ms),
        "fixed_visible_cams": int(fixed_visible),
        "min_fixed_cams_with_cube": int(min_fixed_cams_with_cube),
        "cube_pnp_ok_cams": int(cube_pnp_ok_cams),
        "min_cube_pnp_ok_cams": int(min_cube_pnp_ok_cams),
        "fixed_cube_pnp_ok_cams": int(fixed_cube_pnp_ok_cams),
        "min_fixed_cube_pnp_ok_cams": int(min_fixed_cube_pnp_ok_cams),
        "depth_valid_cams": int(depth_valid_cams),
        "fixed_depth_valid_cams": int(fixed_depth_valid_cams),
        "gripper_markers": int(gripper_markers),
        "gripper_charuco_corners": int(gripper_charuco_corners),
        "min_gripper_charuco_corners": int(min_gripper_charuco_corners),
        "gripper_cube_pnp_ok": bool(gripper_cube_pnp_ok),
        "gripper_depth_valid": bool(gripper_depth_valid),
        "gripper_depth_plane_mean_mm": gripper_depth_plane_mean_mm,
        "per_camera": per_camera,
    }


def build_capture_gate_lines(gate: dict,
                             gripper_cam_idx: Optional[int],
                             frames_dict: Dict[int, dict]) -> List[str]:
    line1 = (
        "SAVE gate: {} | visible cams {}/{} | span {:.1f}/{:.1f} ms".format(
            gate.get("status", "N/A"),
            int(gate.get("cams_with_cube", 0)),
            int(gate.get("min_cams_with_cube", 0)),
            float(gate.get("capture_span_ms", 0.0)),
            float(gate.get("max_capture_span_ms", 0.0)),
        )
    )

    depth_plane = gate.get("gripper_depth_plane_mean_mm")
    depth_plane_txt = "-" if depth_plane is None else "{:.1f}mm".format(float(depth_plane))
    grip_txt = (
        "Gripper: markers={} cube_pnp={} depth={} plane={} charuco={}/{}".format(
            int(gate.get("gripper_markers", 0)),
            "Y" if gate.get("gripper_cube_pnp_ok", False) else "N",
            "Y" if gate.get("gripper_depth_valid", False) else "N",
            depth_plane_txt,
            int(gate.get("gripper_charuco_corners", 0)),
            int(gate.get("min_gripper_charuco_corners", 0)),
        )
    )

    line2 = (
        "{} | fixed visible={}/{}".format(
            grip_txt,
            int(gate.get("fixed_visible_cams", 0)),
            int(gate.get("min_fixed_cams_with_cube", 0)),
        )
    )
    line3 = (
        "PnP quality: total ok cams {}/{} | fixed ok cams {}/{} | depth-valid cams={}".format(
            int(gate.get("cube_pnp_ok_cams", 0)),
            int(gate.get("min_cube_pnp_ok_cams", 0)),
            int(gate.get("fixed_cube_pnp_ok_cams", 0)),
            int(gate.get("min_fixed_cube_pnp_ok_cams", 0)),
            int(gate.get("depth_valid_cams", 0)),
        )
    )

    lines = [line1, line2, line3]
    if not gate.get("pass", False):
        lines.append("FAIL reason: {}".format(gate.get("reason", "unknown")))
    return lines


def append_status_footer(image: np.ndarray,
                         lines: List[str],
                         colors: Optional[List[Tuple[int, int, int]]] = None,
                         bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    if not lines:
        return image
    footer_h = 28 * len(lines) + 12
    footer = np.zeros((footer_h, image.shape[1], 3), dtype=np.uint8)
    footer[:, :] = np.array(bg_color, dtype=np.uint8)
    if colors is None:
        colors = [(255, 255, 255)] * len(lines)
    for idx, line in enumerate(lines):
        color = colors[min(idx, len(colors) - 1)]
        y = 28 + idx * 28
        cv2.putText(footer, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    return cv2.vconcat([image, footer])


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    gripper_cam_idx = m.get("gripper_cam_idx", None)
    return serial_to_idx, gripper_cam_idx, map_path


def load_intrinsics(intr_dir: str, cam_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """카메라 내부 파라미터 행렬 K, 왜곡 계수 D, depth scale을 로드."""
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Intrinsics not found: {p}")
    d = np.load(p, allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64)
    depth_scale = float(d["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in d else 0.001
    if not np.isfinite(depth_scale):
        depth_scale = 0.001
    return K, D, float(depth_scale)


def marker_aspect_ratio(img_pts: np.ndarray) -> float:
    pts = np.asarray(img_pts, dtype=np.float64).reshape(4, 2)
    edge_w = np.linalg.norm(pts[1] - pts[0])
    edge_h = np.linalg.norm(pts[3] - pts[0])
    return float(min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6))


def estimate_per_marker_poses(
    cube: ArucoCubeTarget,
    corners_list: list,
    ids: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_u16: Optional[np.ndarray] = None,
    depth_scale: Optional[float] = None,
) -> List[dict]:
    """
    알려진 큐브 형상을 이용하여 개별 마커의 포즈를 추정.
    마커 1개만으로도 카메라-큐브 변환을 추정할 수 있음.

    마커별 결과 리스트 (rvec, tvec, 재투영 오차 포함)를 반환.
    """
    results = []
    if ids is None or len(ids) == 0:
        return results

    for c, mid in zip(corners_list, ids):
        mid = int(mid)
        if not cube.model.has_marker(mid):
            continue

        img_pts = cube.model.reorder_image_corners(mid, c.reshape(4, 2).astype(np.float64))
        aspect = marker_aspect_ratio(img_pts)
        ippe_candidates = cube.single_marker_ippe_candidates(
            mid,
            c.reshape(4, 2).astype(np.float64),
            K,
            D,
            corners_list=corners_list,
            ids=ids,
            depth_u16=depth_u16,
            depth_scale=depth_scale,
        )
        if not ippe_candidates:
            continue

        pose_candidates = []
        best_idx = None
        best_rank = None
        best_rvec, best_tvec = None, None
        best_T_cam_cube = None
        best_err = None
        best_err_px = None
        best_depth_metrics = None

        for cand in ippe_candidates:
            sol_idx = int(cand["solution_index"])
            rvec = cand["rvec"]
            tvec = cand["tvec"]
            proj = cand["proj2"].reshape(-1, 1, 2)
            err_px = cand["err"]
            err_mean = float(cand["err_mean"])
            T_cam_cube = cand["T_C_O"]
            depth_metrics = cand["depth_metrics"]
            pose_candidates.append({
                "solution_index": int(sol_idx),
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist(),
                "reproj_error_mean_px": err_mean,
                "reproj_error_max_px": float(np.max(err_px)),
                "T_cam_cube_4x4": T_cam_cube.tolist(),
                "z_ok": bool(cand["z_ok"]),
                "vis_ok": bool(cand["vis_ok"]),
                "vis_score": float(cand["vis_score"]),
                "visibility_tier": int(cand["visibility_tier"]),
                **depth_metrics_to_fields(depth_metrics),
            })
            rank = cand["rank"]
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_idx = int(sol_idx)
                best_rvec = rvec
                best_tvec = tvec
                best_T_cam_cube = T_cam_cube
                best_err = err_mean
                best_err_px = err_px
                best_depth_metrics = depth_metrics

        if best_idx is None:
            continue

        results.append({
            "marker_id": mid,
            "face": cube.cfg.id_to_face[mid],
            "corners_2d": img_pts.tolist(),
            "aspect_ratio": aspect,
            "rvec": best_rvec.flatten().tolist(),
            "tvec": best_tvec.flatten().tolist(),
            "reproj_error_mean_px": float(best_err),
            "reproj_error_max_px": float(np.max(best_err_px)),
            "T_cam_cube_4x4": best_T_cam_cube.tolist(),
            "selected_solution_index": int(best_idx),
            "pose_candidates": pose_candidates,
            **depth_metrics_to_fields(best_depth_metrics),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Place-and-Capture calibration: gripper camera + fixed cameras"
    )
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--cube_config_json", type=str, default=None,
                        help="Optional cube config JSON override. Leave unset to use the project's canonical cube definition.")

    # 스트림 설정
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    # 검출 설정
    parser.add_argument("--min_markers", type=int, default=1,
                        help="Min markers per camera to count as 'cube visible'")
    parser.add_argument("--min_cams_with_cube", type=int, default=2,
                        help="Min cameras that must see cube to accept capture")
    parser.add_argument("--min_fixed_cams_with_cube", type=int, default=1,
                        help="Min fixed cameras that must see cube markers to accept capture")
    parser.add_argument("--gripper_cube_min_markers", type=int, default=1,
                        help="Min cube markers required for gripper-camera cube pose")
    parser.add_argument("--gripper_cube_min_aspect", type=float, default=0.35,
                        help="Reject gripper-camera cube markers below this aspect ratio")
    parser.add_argument("--board_mask_pad_px", type=float, default=6.0,
                        help="Extra padding in pixels when masking ChArUco board markers in gripper images")
    parser.add_argument("--min_cube_pnp_ok_cams", type=int, default=2,
                        help="Min cameras with successful cube pose solve to accept capture")
    parser.add_argument("--min_fixed_cube_pnp_ok_cams", type=int, default=1,
                        help="Min fixed cameras with successful cube pose solve to accept capture")
    parser.add_argument("--min_gripper_charuco_corners", type=int, default=8,
                        help="Min ChArUco corners required in gripper camera to accept capture")
    parser.add_argument(
        "--require_gripper_cube_pnp",
        dest="require_gripper_cube_pnp",
        action="store_true",
        default=True,
        help="Require successful cube pose solve in gripper camera (default: on)",
    )
    parser.add_argument(
        "--allow_gripper_cube_pnp_fail",
        dest="require_gripper_cube_pnp",
        action="store_false",
        help="Do not require successful cube pose solve in gripper camera",
    )
    parser.add_argument(
        "--require_gripper_depth_valid",
        dest="require_gripper_depth_valid",
        action="store_true",
        default=True,
        help="Require depth-supported gripper cube pose when depth capture is enabled (default: on)",
    )
    parser.add_argument(
        "--allow_gripper_depth_invalid",
        dest="require_gripper_depth_valid",
        action="store_false",
        help="Allow gripper cube pose even when depth support is invalid",
    )
    parser.add_argument("--max_gripper_depth_plane_mean_mm", type=float, default=15.0,
                        help="Reject a capture when gripper cube depth plane error exceeds this mm (<=0 disables)")
    parser.add_argument("--max_capture_span_ms", type=float, default=120.0,
                        help="Skip a capture when camera timestamps span more than this many ms (<=0 disables)")

    # 뎁스 저장
    parser.add_argument(
        "--save_depth",
        dest="save_depth",
        action="store_true",
        default=True,
        help="Save aligned depth frames for every accepted capture (default: on)",
    )
    parser.add_argument(
        "--no-save-depth",
        dest="save_depth",
        action="store_false",
        help="Disable aligned depth capture and depth PNG saving",
    )

    # 화면 표시
    parser.add_argument("--show", action="store_true")

    # 로봇 모드
    parser.add_argument("--use_robot", action="store_true")
    parser.add_argument("--robot_ip", type=str, default="192.168.0.23")
    parser.add_argument("--robot_port", type=int, default=12348)
    parser.add_argument("--waypoint_file", type=str, default=None,
                        help="JSON file with list of {place, capture} waypoint pairs")
    parser.add_argument("--manual_robot", action="store_true",
                        help="Manual robot mode: server sends capture commands interactively (use with robot_calb.py)")
    parser.add_argument("--settle_time", type=float, default=1.5,
                        help="Wait time (s) after robot signals capture before taking images")
    # start gate
    parser.add_argument("--no_start_gate", action="store_true",
                        help="기본은 cv2 프리뷰 + 'start' 입력 대기. 이 플래그 시 즉시 시작.")

    args = parser.parse_args()

    root = ensure_dir(args.root_folder)
    intr_dir = args.intrinsics_dir
    print(f"[INFO] Depth capture/save: {'ON' if args.save_depth else 'OFF'}")

    # ─── 디바이스 맵 로드 ───
    serial_to_idx, gripper_cam_idx, _ = load_device_map(intr_dir)
    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found.")

    if serial_to_idx is None:
        print("[WARN] No device_map.json. Run Step1 first.")
        serials = sorted(devs.keys())
        idx_serial_pairs = [(i, s) for i, s in enumerate(serials)]
        gripper_cam_idx = None
    else:
        idx_serial_pairs = []
        for serial in devs.keys():
            if serial in serial_to_idx:
                idx_serial_pairs.append((int(serial_to_idx[serial]), serial))
        idx_serial_pairs.sort(key=lambda x: x[0])

    if len(idx_serial_pairs) == 0:
        raise RuntimeError("No usable cameras found.")

    n_fixed = 0
    n_gripper = 0
    print("[INFO] Cameras:")
    for idx, s in idx_serial_pairs:
        if idx == gripper_cam_idx:
            tag = "GRIPPER"
            n_gripper += 1
        else:
            tag = "FIXED"
            n_fixed += 1
        print(f"  cam{idx}: {s} ({tag})")

    if gripper_cam_idx is None:
        print("[WARN] No gripper camera configured in device_map.json.")
        print("[WARN] Gripper camera views will not be available.")
    else:
        print(f"[INFO] Gripper camera: cam{gripper_cam_idx}")

    print(f"[INFO] Fixed cameras: {n_fixed}, Gripper cameras: {n_gripper}")

    # ─── PnP용 내부 파라미터 로드 ───
    cam_intrinsics: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    for ci, _ in idx_serial_pairs:
        try:
            K, D, depth_scale = load_intrinsics(intr_dir, ci)
            cam_intrinsics[ci] = (K, D, depth_scale)
            print(f"[INFO] Loaded intrinsics for cam{ci}")
        except FileNotFoundError:
            print(f"[WARN] No intrinsics for cam{ci}. Per-marker PnP will be skipped.")

    # ─── 카메라 시작 ───
    cams: Dict[int, RealSenseCamera] = {}
    for ci, serial in idx_serial_pairs:
        cam = RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_color=True,
            use_depth=args.save_depth,
            align_depth_to_color=True,
            warmup_frames=10,
        )
        cam.start()
        cams[ci] = cam
        ensure_dir(os.path.join(root, f"cam{ci}"))

    cfg, cube_cfg_source = resolve_cube_config_for_run(
        root_folder=root,
        cube_config_json=args.cube_config_json,
        default_cfg=get_default_cube_config(),
    )
    cube = ArucoCubeTarget(cfg)
    _cube_ids = set(cfg.marker_ids)  # {0,1,2,3,4} — filter out board markers
    print(f"[INFO] Cube config source: {cube_cfg_source}")
    print(f"[INFO] Cube id_to_face: {cfg.id_to_face}")

    # ChArUco board target — gripper camera only
    charuco_cfg = CharucoBoardConfig()
    charuco = CharucoTarget(charuco_cfg)
    print(f"[INFO] ChArUco board: {charuco_cfg.squares_x}x{charuco_cfg.squares_y}, "
          f"marker_id_start={charuco_cfg.marker_id_start}")

    if not args.save_depth and args.require_gripper_depth_valid:
        print("[WARN] Depth capture is disabled; gripper depth-valid gate will be ignored.")
        args.require_gripper_depth_valid = False
    capture_gate_cfg = make_capture_gate_config(args)
    print("[INFO] Capture gate:")
    print("  visible cams >= {} | fixed visible >= {} | cube_pnp ok cams >= {} | fixed cube_pnp ok cams >= {}".format(
        capture_gate_cfg["min_cams_with_cube"],
        capture_gate_cfg["min_fixed_cams_with_cube"],
        capture_gate_cfg["min_cube_pnp_ok_cams"],
        capture_gate_cfg["min_fixed_cube_pnp_ok_cams"],
    ))
    print("  gripper cube_pnp required={} | gripper charuco >= {} | gripper depth required={} | depth plane <= {:.1f}mm | span <= {:.1f}ms".format(
        "yes" if capture_gate_cfg["require_gripper_cube_pnp"] else "no",
        capture_gate_cfg["min_gripper_charuco_corners"],
        "yes" if capture_gate_cfg["require_gripper_depth_valid"] else "no",
        capture_gate_cfg["max_gripper_depth_plane_mean_mm"],
        capture_gate_cfg["max_capture_span_ms"],
    ))
    print(f"  gripper board mask pad: {float(args.board_mask_pad_px):.1f}px")

    # (Board marker detection uses charuco.detect() directly — no separate detector needed)

    # ─── 웨이포인트 로드 ───
    waypoint_list: List[dict] = []
    waypoint_bundle: dict = {}
    if args.waypoint_file:
        with open(args.waypoint_file, "r") as f:
            waypoint_payload = json.load(f)
        waypoint_list, waypoint_bundle = normalize_waypoint_file_payload(waypoint_payload)
        print(f"[INFO] Loaded {len(waypoint_list)} waypoints from {args.waypoint_file}")

    # ─── 로봇 클라이언트 ───
    # start 게이트 전에 연결을 끝내서 cv2 창이 응답 없음 상태가 되지 않도록 한다.
    robot_client: Optional[PlaceCaptureClient] = None
    manual_sock = None
    if args.use_robot and not args.manual_robot:
        robot_client = PlaceCaptureClient(args.robot_ip, args.robot_port)
        robot_client.connect()
    elif args.use_robot and args.manual_robot:
        import socket as _sock
        manual_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        manual_sock.settimeout(None)
        print(f"[ManualRobot] Connecting to {args.robot_ip}:{args.robot_port} ...")
        manual_sock.connect((args.robot_ip, args.robot_port))
        print(f"[ManualRobot] Connected to {args.robot_ip}:{args.robot_port}")

    # ─── 메타 데이터 (기존 meta.json이 있으면 이어서 저장) ───
    meta_path = os.path.join(root, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta_cfg, meta_cfg_source = load_cube_config_from_meta(root, default_cfg=cfg)
        if meta.get("captures") and not cube_configs_equivalent(meta_cfg, cfg):
            mismatch_keys = cube_config_mismatch_keys(cfg, meta_cfg)
            raise RuntimeError(
                "Existing meta.json uses a different cube definition.\n"
                f"Resolved cube config: {cube_cfg_source}\n"
                f"Session cube config: {meta.get('cube_config_source', meta_cfg_source)}\n"
                f"Differing fields: {', '.join(mismatch_keys) if mismatch_keys else 'unknown'}\n"
                "Use a new session folder, or run recompute_session_cube_pnp.py with the intended cube config before resuming."
            )
        event_id = max((int(c.get("event_id", -1)) for c in meta.get("captures", [])), default=-1) + 1
        print(f"[INFO] Resuming from existing meta.json ({len(meta['captures'])} captures, next event_id={event_id})")
    else:
        meta = {
            "root_folder": os.path.abspath(root),
            "gripper_cam_idx": gripper_cam_idx,
            "n_fixed_cams": n_fixed,
            "n_gripper_cams": n_gripper,
            "cam_indices": [ci for ci, _ in idx_serial_pairs],
            "cube_config_source": cube_cfg_source,
            "cube_config": cube_config_to_dict(cfg),
            "captures": [],
        }
        event_id = 0
        print("[INFO] New session (meta.json created)")
    meta["cube_config_source"] = cube_cfg_source
    if "cube_config" not in meta:
        meta["cube_config"] = cube_config_to_dict(cfg)
    else:
        meta["cube_config"] = cube_config_to_dict(cfg)
    quad_dir = ensure_dir(os.path.join(root, "marker_quads"))
    cam_order = sorted(ci for ci, _ in idx_serial_pairs)

    def build_frame_record(
        ci: int,
        color: np.ndarray,
        depth: Optional[np.ndarray],
        ts_ms: Optional[float],
        include_marker_poses: bool = True,
        include_charuco_pose: bool = True,
        log_pose_status: bool = False,
    ) -> dict:
        detect_info = detect_cube_markers_in_frame(
            color,
            cube,
            cube_ids=cfg.marker_ids,
            charuco=charuco if ci == gripper_cam_idx else None,
            is_gripper=(ci == gripper_cam_idx),
            board_mask_pad_px=float(args.board_mask_pad_px),
        )
        corners = detect_info["corners"]
        ids = detect_info["ids"]
        cube_img = detect_info["cube_image"]
        n_markers = 0 if ids is None else len(ids)
        fr = {
            "color": color,
            "depth": depth,
            "ts_ms": ts_ms,
            "ok": bool(n_markers >= args.min_markers),
            "n_markers": n_markers,
            "ids": ([] if ids is None else [int(x) for x in ids]),
            "corners": corners,
            "ids_np": ids,
            "marker_poses": [],
            "cube_pnp": None,
            "cube_detect_raw_ids": detect_info["raw_ids"],
            "cube_detect_filtered_ids": detect_info["filtered_ids"],
            "board_mask_applied": bool(detect_info["board_mask_applied"]),
        }
        if ci == gripper_cam_idx:
            fr["board_mkr_corners"] = detect_info["board_mkr_corners"]
            fr["board_mkr_ids"] = detect_info["board_mkr_ids"]
            fr["ch_corners"] = detect_info["ch_corners"]
            fr["ch_ids"] = detect_info["ch_ids"]
            fr["charuco_detect_n"] = int(detect_info["charuco_detect_n"])

        intr = cam_intrinsics.get(ci)
        if intr is not None and ids is not None and len(ids) > 0:
            K, D, depth_scale = intr
            if include_marker_poses:
                fr["marker_poses"] = estimate_per_marker_poses(
                    cube, corners, ids, K, D,
                    depth_u16=depth, depth_scale=depth_scale)

            min_cube_markers = args.gripper_cube_min_markers if ci == gripper_cam_idx else 1
            min_cube_aspect = args.gripper_cube_min_aspect if ci == gripper_cam_idx else 0.0
            pnp_ok, rvec, tvec, used_ids, reproj = cube.solve_pnp_cube(
                cube_img, K, D,
                use_ransac=True,
                min_markers=max(int(min_cube_markers), 1),
                return_reproj=True,
                min_aspect=float(min_cube_aspect),
                depth_u16=depth,
                depth_scale=depth_scale,
            )
            tag = "G" if ci == gripper_cam_idx else "F"
            if log_pose_status:
                if pnp_ok:
                    print(f"  [PnP] cam{ci}({tag}): OK ids={used_ids} reproj={reproj['err_mean']:.2f}px")
                else:
                    det_ids = [int(x) for x in ids] if ids is not None else []
                    print(
                        f"  [PnP] cam{ci}({tag}): FAILED "
                        f"(cube_ids={det_ids}, raw_ids={detect_info['raw_ids']}, mask={fr['board_mask_applied']})"
                    )
            if pnp_ok and rvec is not None:
                T_cam_cube = rodrigues_to_Rt(rvec, tvec)
                fr["cube_pnp"] = {
                    "ok": True,
                    "rvec": rvec.flatten().tolist(),
                    "tvec": tvec.flatten().tolist(),
                    "used_ids": [int(x) for x in used_ids],
                    "reproj_mean_px": reproj["err_mean"] if reproj else None,
                    "T_cam_cube_4x4": T_cam_cube.tolist(),
                    "min_markers_required": int(min_cube_markers),
                    "min_aspect_required": float(min_cube_aspect),
                    **depth_metrics_to_fields((reproj or {}).get("depth_metrics")),
                }

        if ci == gripper_cam_idx:
            if include_charuco_pose and intr is not None:
                K, D, _ = intr
                try:
                    ch_ok, ch_rvec, ch_tvec, ch_n, ch_reproj = charuco.estimate_pose(color, K, D)
                except Exception as e:
                    ch_ok = False
                    ch_n = int(fr.get("charuco_detect_n", 0))
                    ch_reproj = None
                    if log_pose_status:
                        print(f"  [ChArUco] pose ERROR: {e}")
                if ch_ok and ch_rvec is not None:
                    T_cam_board = rodrigues_to_Rt(ch_rvec, ch_tvec)
                    fr["charuco"] = {
                        "ok": True,
                        "n_corners": int(ch_n),
                        "reproj_error_px": float(ch_reproj) if ch_reproj is not None else None,
                        "rvec": ch_rvec.flatten().tolist(),
                        "tvec": ch_tvec.flatten().tolist(),
                        "T_cam_board_4x4": T_cam_board.tolist(),
                    }
                    if log_pose_status:
                        print(f"  [ChArUco] OK: {ch_n} corners, reproj={ch_reproj:.3f}px")
                elif log_pose_status:
                    print(f"  [ChArUco] FAILED (corners={int(fr.get('charuco_detect_n', 0))})")

        return fr

    # ── start 게이트: 첫 cv2 프리뷰 + 'start' 입력 대기 ──
    if not args.no_start_gate:
        extra = []
        if args.use_robot:
            extra.append(f"robot {args.robot_ip}:{args.robot_port}"
                         + (" (manual)" if args.manual_robot else ""))
        if waypoint_list:
            extra.append(f"{len(waypoint_list)} waypoints loaded")
        if not wait_for_start_command_capture(cams, cam_order, gripper_cam_idx, extra):
            for cam in cams.values():
                cam.stop()
            cv2.destroyAllWindows()
            return

    print("\nControls:")
    if args.use_robot and waypoint_list and not args.manual_robot:
        print("  ENTER : move robot to next saved waypoint")
        print("  SPACE : capture at current robot pose")
    else:
        print("  SPACE : manual capture (if in manual mode)")
    print("  ESC/q : quit\n")

    def do_capture(
        capture_pose_6dof: Optional[List[float]] = None,
        place_pose_6dof: Optional[List[float]] = None,
        pose_index: Optional[int] = None,
        grip_target_tvec: Optional[List[float]] = None,
        robot_joints_6dof: Optional[List[float]] = None,
        set_cube_center_6dof: Optional[List[float]] = None,
        set_index: Optional[int] = None,
    ) -> Tuple[bool, dict]:
        """모든 카메라에서 마커별 포즈 추정과 함께 촬영."""
        nonlocal event_id

        # 안정화 대기
        if args.settle_time > 0 and args.use_robot:
            time.sleep(args.settle_time)

        frames: Dict[int, dict] = {}

        # Software-sync: 각 카메라의 latest ts 중 가장 오래된 것(=가장 느린 카메라)을
        # 기준으로 잡고, 다른 카메라들은 버퍼에서 그 시각에 가장 가까운 프레임을 고른다.
        # 하드웨어 sync 없는 RealSense들의 timestamp span을 1프레임(~33ms) 이내로 좁힘.
        latest_ts_list = []
        for ci, cam in cams.items():
            _c, _d, ts_ms = cam.get_latest()
            if ts_ms is not None:
                latest_ts_list.append(ts_ms)

        if latest_ts_list:
            target_ts = min(latest_ts_list)
            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_at(target_ts)
                if color is None:
                    continue
                frames[ci] = build_frame_record(
                    ci, color, depth, ts_ms,
                    include_marker_poses=True,
                    include_charuco_pose=True,
                    log_pose_status=True,
                )
        else:
            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_latest()
                if color is None:
                    continue
                frames[ci] = build_frame_record(
                    ci, color, depth, ts_ms,
                    include_marker_poses=True,
                    include_charuco_pose=True,
                    log_pose_status=True,
                )

        gate = evaluate_capture_gate(
            frames,
            capture_gate_cfg,
            gripper_cam_idx=gripper_cam_idx,
        )
        capture_span_ms = float(gate["capture_span_ms"])
        if not gate["pass"]:
            print(f"[SKIP] {gate['reason']}")
            return False, gate

        # ─── 저장 ───
        fid = int(event_id)
        cap_rec: dict = {
            "event_id": fid,
            "pose_index": pose_index,
            "capture_span_ms": float(capture_span_ms),
            "capture_gate": gate,
            "cams": {},
        }

        # 로봇 포즈 데이터
        # capture_pose = 이미지 촬영 시 현재 로봇 TCP
        # Step3에서 참조: robot_pose_6dof / robot_pose_matrix_4x4
        robot_tcp = capture_pose_6dof or place_pose_6dof
        if robot_tcp is not None:
            tcp_f = [float(x) for x in robot_tcp]
            cap_rec["robot_pose_6dof"] = tcp_f        # Step3 compatible
            cap_rec["capture_pose_6dof"] = tcp_f      # new2 format
            try:
                T44 = euler_deg_to_matrix(*tcp_f).tolist()
                cap_rec["robot_pose_matrix_4x4"] = T44  # Step3 compatible
                cap_rec["capture_pose_matrix_4x4"] = T44
            except Exception:
                pass

        if grip_target_tvec is not None:
            cap_rec["grip_target_tvec"] = [float(x) for x in grip_target_tvec]

        if robot_joints_6dof is not None:
            cap_rec["robot_joints_6dof"] = [float(x) for x in robot_joints_6dof]

        if set_cube_center_6dof is not None:
            cap_rec["set_cube_center_6dof"] = [float(x) for x in set_cube_center_6dof]

        if set_index is not None:
            cap_rec["set_index"] = set_index

        if place_pose_6dof is not None and place_pose_6dof != robot_tcp:
            cap_rec["place_pose_6dof"] = [float(x) for x in place_pose_6dof]
            try:
                cap_rec["place_pose_matrix_4x4"] = euler_deg_to_matrix(
                    *place_pose_6dof
                ).tolist()
            except Exception:
                pass

        for ci in sorted(frames.keys()):
            fr = frames[ci]

            rgb_rel = f"cam{ci}/rgb_{fid:05d}.jpg"
            cv2.imwrite(os.path.join(root, rgb_rel), fr["color"])

            depth_rel = None
            if args.save_depth and fr["depth"] is not None:
                depth_rel = f"cam{ci}/depth_{fid:05d}.png"
                cv2.imwrite(os.path.join(root, depth_rel), fr["depth"])

            cam_rec = {
                "saved": True,
                "is_gripper": (ci == gripper_cam_idx),
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "ts_ms": fr["ts_ms"],
                "n_markers_detected": fr["n_markers"],
                "marker_ids": fr["ids"],
                "cube_visible": fr["ok"],
                "markers": fr["marker_poses"],  # per-marker PnP results
                "cube_detect_raw_ids": fr.get("cube_detect_raw_ids", []),
                "cube_detect_filtered_ids": fr.get("cube_detect_filtered_ids", []),
                "board_mask_applied": bool(fr.get("board_mask_applied", False)),
            }
            if ci == gripper_cam_idx:
                cam_rec["charuco_detect_n"] = int(fr.get("charuco_detect_n", 0))

            if fr["cube_pnp"] is not None:
                cam_rec["cube_pnp"] = fr["cube_pnp"]

            if ci == gripper_cam_idx and fr.get("charuco") is not None:
                cam_rec["charuco"] = fr["charuco"]

            cap_rec["cams"][str(ci)] = cam_rec

        meta["captures"].append(cap_rec)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 마커 오버레이가 포함된 2x2 분할 이미지 저장
        quad = make_quad_image(frames, cam_order, cube, gripper_cam_idx)
        quad_path = os.path.join(quad_dir, f"frame_{fid:05d}.jpg")
        cv2.imwrite(quad_path, quad)

        # (프리뷰 스레드와 충돌 방지: imshow 제거, 파일로만 저장)

        # 요약 출력
        cam_summary = []
        for ci in sorted(frames.keys()):
            fr = frames[ci]
            tag = "G" if ci == gripper_cam_idx else "F"
            n = fr["n_markers"]
            cam_summary.append(f"cam{ci}({tag}):{n}mkr")
        charuco_txt = ""
        gi_rec = cap_rec["cams"].get(str(gripper_cam_idx), {})
        if "charuco" in gi_rec:
            ch = gi_rec["charuco"]
            charuco_txt = f" charuco={ch['n_corners']}cor"
        print(f"[SAVE] event={fid} | {' '.join(cam_summary)} span={capture_span_ms:.1f}ms{charuco_txt}")
        event_id += 1
        return True, gate

    try:
        if args.use_robot and args.manual_robot:
            # ─── 수동 로봇 모드 (robot_calb.py 서버 사용) ───
            # cv2는 main thread 전용. 소켓 recv는 백그라운드 스레드.
            # main thread가 recv에 블로킹되면 cv2 윈도우가 응답 없음 상태가 되므로
            # 분리한다.
            print("[MODE] Manual Robot - waiting for server capture commands")
            print("[INFO] Move robot on server side, press 'c' to capture\n")

            import threading

            # Waypoint accumulator (mirror of robot's capture_waypoints.json)
            wp_list: list = []
            wp_set_joints = None
            wp_set_tcp = None
            wp_set_cube_center = None

            network_done = threading.Event()
            user_quit = threading.Event()

            # 짧은 timeout으로 recv가 주기적으로 깨어나 종료 플래그를 확인하게 함.
            manual_sock.settimeout(0.5)

            def network_loop():
                nonlocal wp_set_joints, wp_set_tcp, wp_set_cube_center
                try:
                    while not network_done.is_set() and not user_quit.is_set():
                        try:
                            data = manual_sock.recv(8192)
                        except _sock.timeout:
                            continue
                        except OSError as e:
                            # 정상 종료 시 main thread가 socket을 닫아서 EBADF가 뜸 → 무시
                            if not (user_quit.is_set() or network_done.is_set()):
                                print(f"[ManualRobot] socket error: {e}")
                            break
                        if not data:
                            print("[ManualRobot] Server disconnected.")
                            break

                        try:
                            msg = json.loads(data.decode("utf-8").strip())
                        except Exception as e:
                            print(f"[ManualRobot] JSON parse error: {e}")
                            continue

                        cmd = msg.get("command", "")
                        if cmd == "quit":
                            print("[ManualRobot] Server sent quit.")
                            break

                        if cmd == "capture":
                            capture_tcp = msg.get("capture_pose_6dof")
                            pose_idx = msg.get("pose_index", event_id)
                            g_tvec = msg.get("grip_target_tvec")
                            r_joints = msg.get("robot_joints_6dof")
                            s_cube = msg.get("set_cube_center_6dof")
                            s_idx = msg.get("set_index")
                            m_set_joints = msg.get("set_joints")
                            m_set_tcp = msg.get("set_tcp")
                            m_place_joints = msg.get("place_joints")

                            print(f"\n[ManualRobot] Capture signal received (pose_index={pose_idx}, set_index={s_idx})")
                            if capture_tcp:
                                print(f"  TCP: {capture_tcp}")
                            if r_joints:
                                print(f"  Joints: {r_joints}")

                            saved, gate = do_capture(
                                capture_pose_6dof=capture_tcp,
                                pose_index=pose_idx,
                                grip_target_tvec=g_tvec,
                                robot_joints_6dof=r_joints,
                                set_cube_center_6dof=s_cube,
                                set_index=s_idx,
                            )

                            status = "success" if saved else "skipped"
                            resp = json.dumps({
                                "action": "captured",
                                "status": status,
                                "reason": gate.get("reason"),
                            })
                            try:
                                manual_sock.sendall(resp.encode("utf-8"))
                            except OSError:
                                break

                            if m_set_joints is not None:
                                wp_set_joints = m_set_joints
                            if m_set_tcp is not None:
                                wp_set_tcp = m_set_tcp
                            if s_cube is not None:
                                wp_set_cube_center = s_cube

                            wp_entry = {
                                "pose_index": pose_idx,
                                "capture_joints": r_joints,
                                "capture_tcp": capture_tcp,
                                "cube_center_6dof": msg.get("cube_center_pose_6dof"),
                                "set_index": s_idx,
                            }
                            if m_place_joints is not None:
                                wp_entry["place_joints"] = m_place_joints
                            wp_list.append(wp_entry)

                            if saved:
                                print(f"[OK] Capture {pose_idx} saved")
                            else:
                                print(f"[SKIP] Capture {pose_idx} skipped")

                        elif cmd == "detect":
                            if gripper_cam_idx is None or gripper_cam_idx not in cams:
                                resp = json.dumps({"ok": False, "reason": "no_gripper_cam"})
                                try:
                                    manual_sock.sendall(resp.encode("utf-8"))
                                except OSError:
                                    break
                                continue
                            if gripper_cam_idx not in cam_intrinsics:
                                resp = json.dumps({"ok": False, "reason": "no_intrinsics"})
                                try:
                                    manual_sock.sendall(resp.encode("utf-8"))
                                except OSError:
                                    break
                                continue

                            g_color, g_depth, _ = cams[gripper_cam_idx].get_latest()
                            if g_color is None:
                                resp = json.dumps({"ok": False, "reason": "no_image"})
                                try:
                                    manual_sock.sendall(resp.encode("utf-8"))
                                except OSError:
                                    break
                                continue

                            g_K, g_D, g_depth_scale = cam_intrinsics[gripper_cam_idx]
                            detect_info = detect_cube_markers_in_frame(
                                g_color, cube,
                                cube_ids=cfg.marker_ids,
                                charuco=charuco,
                                is_gripper=True,
                                board_mask_pad_px=float(args.board_mask_pad_px),
                            )
                            det_ok, det_rv, det_tv, det_used = cube.solve_pnp_cube(
                                detect_info["cube_image"], g_K, g_D, use_ransac=False, min_markers=1,
                                reproj_thr_mean_px=10.0,
                                min_aspect=float(args.gripper_cube_min_aspect),
                                depth_u16=g_depth, depth_scale=g_depth_scale)

                            if det_ok:
                                resp = json.dumps({
                                    "ok": True,
                                    "tvec": det_tv.flatten().tolist(),
                                    "rvec": det_rv.flatten().tolist(),
                                    "used_ids": [int(x) for x in det_used],
                                })
                                print(f"[Detect] tvec=[{det_tv[0][0]:.4f}, {det_tv[1][0]:.4f}, {det_tv[2][0]:.4f}] ids={det_used}")
                            else:
                                resp = json.dumps({"ok": False, "reason": "detection_failed",
                                                   "n_markers": len(det_used) if det_used else 0})
                                print(
                                    f"[Detect] Failed (cube_ids={det_used}, "
                                    f"raw_ids={detect_info['raw_ids']}, mask={detect_info['board_mask_applied']})"
                                )
                            try:
                                manual_sock.sendall(resp.encode("utf-8"))
                            except OSError:
                                break
                        else:
                            print(f"[ManualRobot] Unknown command: {cmd}")
                except Exception as e:
                    import traceback
                    print(f"[ManualRobot] network thread crashed: {e}")
                    traceback.print_exc()
                finally:
                    network_done.set()

            net_thread = threading.Thread(target=network_loop, daemon=True)
            net_thread.start()
            if args.show:
                print("[INFO] Live preview started (4-camera quad view)")

            try:
                # Main thread: cv2 preview만 담당. recv는 net_thread.
                while not network_done.is_set():
                    if args.show:
                        live_frames = {}
                        for ci, cam in cams.items():
                            color, depth, ts_ms = cam.get_latest()
                            if color is None:
                                continue
                            live_frames[ci] = build_frame_record(
                                ci, color, depth, ts_ms,
                                include_marker_poses=False,
                                include_charuco_pose=False,
                                log_pose_status=False,
                            )

                        if live_frames:
                            quad = make_quad_image(live_frames, cam_order, cube, gripper_cam_idx)
                            gate = evaluate_capture_gate(
                                live_frames,
                                capture_gate_cfg,
                                gripper_cam_idx=gripper_cam_idx,
                            )
                            gate_lines = build_capture_gate_lines(gate, gripper_cam_idx, live_frames)
                            gate_colors = [(0, 255, 0)] if gate["pass"] else [(0, 0, 255)]
                            gate_colors = gate_colors + [(255, 255, 255)] * (len(gate_lines) - 1)
                            quad = append_status_footer(quad, gate_lines, gate_colors)
                            ph = int(quad.shape[0] * 0.6)
                            pw = int(quad.shape[1] * 0.6)
                            preview = cv2.resize(quad, (pw, ph))
                            cv2.imshow("Capture Preview", preview)

                        key = cv2.waitKey(50) & 0xFF
                        if key == 27 or key == ord('q'):
                            print("[ManualRobot] User quit preview.")
                            user_quit.set()
                            break
                    else:
                        time.sleep(0.1)

            finally:
                network_done.set()
                user_quit.set()
                try:
                    manual_sock.shutdown(_sock.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    manual_sock.close()
                except Exception:
                    pass
                net_thread.join(timeout=2.0)

                # Save capture_waypoints.json (mirror of robot server's file)
                if wp_list:
                    wp_save = {
                        "set_joints": wp_set_joints,
                        "set_tcp": wp_set_tcp,
                        "set_cube_center": wp_set_cube_center,
                        "waypoints": wp_list,
                    }
                    wp_path = os.path.join(root, "capture_waypoints.json")
                    with open(wp_path, "w") as f:
                        json.dump(wp_save, f, indent=2)
                    print(f"[INFO] Waypoints saved: {wp_path} ({len(wp_list)} poses)")

            print(f"\n[DONE] Manual robot capture complete. {event_id} captures saved.")

        elif args.use_robot and waypoint_list:
            # ─── Robot waypoint step mode (ENTER=move, SPACE=capture) ───
            print("[MODE] Robot waypoint step mode")
            print(f"[INFO] {len(waypoint_list)} waypoints loaded")
            print("[INFO] Press ENTER to move to the next waypoint, then SPACE to save the capture.\n")

            waypoint_cursor = 0
            pending_capture = None
            status_line = "Press ENTER to move the robot to waypoint 1."

            def build_live_preview_frames() -> Dict[int, dict]:
                live_frames: Dict[int, dict] = {}
                for ci, cam in cams.items():
                    color, depth, ts_ms = cam.get_latest()
                    if color is None:
                        continue
                    live_frames[ci] = build_frame_record(
                        ci, color, depth, ts_ms,
                        include_marker_poses=False,
                        include_charuco_pose=False,
                        log_pose_status=False,
                    )
                return live_frames

            try:
                while True:
                    live_frames = build_live_preview_frames()
                    quad = make_quad_image(live_frames, cam_order, cube, gripper_cam_idx)
                    gate = evaluate_capture_gate(
                        live_frames,
                        capture_gate_cfg,
                        gripper_cam_idx=gripper_cam_idx,
                    )
                    gate_lines = build_capture_gate_lines(gate, gripper_cam_idx, live_frames)
                    footer_h = 28 * (2 + len(gate_lines)) + 16
                    footer = np.zeros((footer_h, quad.shape[1], 3), dtype=np.uint8)
                    if waypoint_cursor < len(waypoint_list):
                        next_pose_index = waypoint_list[waypoint_cursor].get("pose_index", waypoint_cursor)
                        next_set_index = waypoint_list[waypoint_cursor].get("set_index")
                        next_txt = f"Next waypoint: idx={waypoint_cursor + 1}/{len(waypoint_list)} pose_index={next_pose_index} set_index={next_set_index}"
                    else:
                        next_txt = f"Next waypoint: completed ({len(waypoint_list)}/{len(waypoint_list)})"
                    cv2.putText(footer, next_txt, (12, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                    cv2.putText(footer, status_line, (12, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
                    for line_idx, gate_line in enumerate(gate_lines):
                        color = (0, 255, 0) if (line_idx == 0 and gate["pass"]) else (
                            (0, 0, 255) if line_idx == 0 else (255, 255, 255)
                        )
                        cv2.putText(
                            footer,
                            gate_line,
                            (12, 84 + line_idx * 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.56,
                            color,
                            2,
                        )
                    preview = cv2.vconcat([quad, footer])
                    cv2.imshow("Capture Preview", preview)

                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord('q')):
                        if pending_capture is not None:
                            try:
                                robot_client.send_captured("aborted", reason="user quit before capture")
                            except Exception:
                                pass
                        break

                    if key in (10, 13):
                        if pending_capture is not None:
                            status_line = "Capture is pending. Press SPACE to save before moving again."
                            print("[WARN] Capture is pending. Press SPACE before requesting the next waypoint.")
                            continue
                        if waypoint_cursor >= len(waypoint_list):
                            status_line = "All waypoints completed. Press q to quit."
                            print("[INFO] All waypoints have already been processed.")
                            continue

                        wp = waypoint_list[waypoint_cursor]
                        try:
                            place_pose, capture_pose, place_kind, capture_kind = resolve_waypoint_motion(
                                wp, waypoint_bundle)
                        except Exception as e:
                            status_line = f"Waypoint parse failed: {e}"
                            print(f"[ERROR] Failed to parse waypoint {waypoint_cursor + 1}: {e}")
                            continue

                        extra_fields = {}
                        if wp.get("pose_index") is not None:
                            extra_fields["pose_index"] = int(wp["pose_index"])
                        if wp.get("set_index") is not None:
                            extra_fields["set_index"] = int(wp["set_index"])
                        if _as_pose6(wp.get("capture_tcp")) is not None:
                            extra_fields["capture_tcp"] = _as_pose6(wp.get("capture_tcp"))
                        if _as_pose6(wp.get("cube_center_6dof")) is not None:
                            extra_fields["cube_center_6dof"] = _as_pose6(wp.get("cube_center_6dof"))

                        print(f"\n[Robot] Move request {waypoint_cursor + 1}/{len(waypoint_list)}")
                        print(f"  pose_index: {wp.get('pose_index', waypoint_cursor)} set_index: {wp.get('set_index')}")
                        print(f"  place ({place_kind}):   {place_pose}")
                        print(f"  capture ({capture_kind}): {capture_pose}")

                        try:
                            ok, actual_capture_tcp, actual_place_tcp = robot_client.run_single_waypoint(
                                place_pose,
                                capture_pose,
                                place_kind=place_kind,
                                capture_kind=capture_kind,
                                extra_fields=extra_fields,
                            )
                        except Exception as e:
                            print(f"[ERROR] Robot communication error: {e}")
                            break

                        if not ok:
                            print("[INFO] Robot quit or error.")
                            break

                        pending_capture = {
                            "waypoint": wp,
                            "actual_capture_tcp": actual_capture_tcp,
                            "actual_place_tcp": actual_place_tcp,
                        }
                        status_line = (
                            f"Robot is at pose_index={wp.get('pose_index', waypoint_cursor)}. "
                            "Press SPACE to capture."
                        )
                        print("[INFO] Robot reached capture pose. Press SPACE to save images.")
                        continue

                    if key == 32:  # SPACE
                        if pending_capture is None:
                            status_line = "No robot pose is active. Press ENTER first."
                            print("[WARN] No active waypoint. Press ENTER to move the robot first.")
                            continue

                        wp = pending_capture["waypoint"]
                        capture_tcp = pending_capture["actual_capture_tcp"] or _as_pose6(wp.get("capture_tcp"))
                        place_tcp = pending_capture["actual_place_tcp"]
                        saved, gate = do_capture(
                            capture_pose_6dof=capture_tcp,
                            place_pose_6dof=place_tcp,
                            pose_index=int(wp.get("pose_index", waypoint_cursor)),
                            robot_joints_6dof=_as_pose6(wp.get("capture_joints")),
                            set_cube_center_6dof=(
                                _as_pose6(wp.get("cube_center_6dof")) or
                                _as_pose6(waypoint_bundle.get("set_cube_center"))
                            ),
                            set_index=wp.get("set_index"),
                        )

                        capture_status = "success" if saved else "skipped"
                        try:
                            robot_client.send_captured(capture_status, reason=gate.get("reason"))
                        except Exception as e:
                            print(f"[ERROR] Failed to acknowledge capture: {e}")
                            break

                        if saved:
                            print(f"[OK] Waypoint {waypoint_cursor + 1} captured successfully")
                            status_line = f"Capture saved for waypoint {waypoint_cursor + 1}. Press ENTER for the next waypoint."
                        else:
                            print(f"[WARN] Waypoint {waypoint_cursor + 1} capture skipped")
                            status_line = (
                                f"Capture skipped for waypoint {waypoint_cursor + 1}: "
                                f"{gate.get('reason', 'unknown')}"
                            )

                        waypoint_cursor += 1
                        pending_capture = None

            finally:
                try:
                    robot_client.wait_for_ready()
                    robot_client.send_quit()
                except Exception:
                    pass

            print(f"\n[DONE] Robot waypoint step capture complete. {event_id} captures saved.")

        else:
            # ─── Manual mode ───
            print("[MODE] Manual capture (press SPACE)")
            while True:
                frames_view: Dict[int, dict] = {}
                for ci, cam in cams.items():
                    color, depth, ts_ms = cam.get_latest()
                    if color is None:
                        continue
                    frames_view[ci] = build_frame_record(
                        ci, color, depth, ts_ms,
                        include_marker_poses=False,
                        include_charuco_pose=False,
                        log_pose_status=False,
                    )

                if args.show:
                    gate = evaluate_capture_gate(
                        frames_view,
                        capture_gate_cfg,
                        gripper_cam_idx=gripper_cam_idx,
                    )
                    gate_lines = build_capture_gate_lines(gate, gripper_cam_idx, frames_view)
                    panel = np.zeros((28 * len(gate_lines) + 12, 1100, 3), dtype=np.uint8)
                    for line_idx, gate_line in enumerate(gate_lines):
                        color = (0, 255, 0) if (line_idx == 0 and gate["pass"]) else (
                            (0, 0, 255) if line_idx == 0 else (255, 255, 255)
                        )
                        cv2.putText(panel, gate_line, (12, 28 + line_idx * 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                    cv2.imshow("Capture Gate", panel)
                    for ci in sorted(frames_view.keys()):
                        img = frames_view[ci]["color"].copy()
                        ids_np = frames_view[ci]["ids_np"]
                        corners = frames_view[ci]["corners"]
                        if ids_np is not None:
                            try:
                                draw_ids = ids_np.reshape(-1, 1) if getattr(ids_np, "ndim", 1) == 1 else ids_np
                                cv2.aruco.drawDetectedMarkers(img, corners, draw_ids)
                            except Exception:
                                pass
                        tag = "GRIP" if ci == gripper_cam_idx else "FIX"
                        n = 0 if ids_np is None else len(ids_np)
                        txt = f"cam{ci}({tag}) markers={n} ok={frames_view[ci]['ok']}"
                        cv2.putText(img, txt, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow(f"cam{ci}", img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                if key == 32:  # SPACE
                    do_capture()

    finally:
        for cam in cams.values():
            cam.stop()
        if robot_client:
            robot_client.close()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Total captures: {event_id}")
    print(f"  Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
