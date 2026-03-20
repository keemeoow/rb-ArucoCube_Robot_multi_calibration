#!/usr/bin/env python3
"""
Create per-frame 2x2 quad images (fixed cameras x3 + gripper x1) with ArUco marker overlays.

Examples:
  python make_marker_quad_images.py --session_root ./data/session_manual
  python make_marker_quad_images.py --data_root ./data
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget
from config import CubeConfig


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_frame_id_from_rgb_path(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.match(r"rgb_(\d+)\.(jpg|jpeg|png)$", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def discover_sessions(data_root: str) -> List[str]:
    sessions = []
    if not os.path.isdir(data_root):
        return sessions
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "meta.json")):
            sessions.append(p)
    return sessions


def load_meta(session_root: str) -> Dict:
    meta_path = os.path.join(session_root, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def choose_cam_order(meta: Dict, session_root: str) -> Tuple[List[int], Optional[int]]:
    """
    Return exactly 4 camera indices:
      [fixed_1, fixed_2, fixed_3, gripper]
    If data is incomplete, missing slots are filled from available cams.
    """
    gripper_idx = meta.get("gripper_cam_idx", None)
    cam_indices = meta.get("cam_indices", None)

    if not cam_indices:
        cam_indices = []
        for p in glob.glob(os.path.join(session_root, "cam*")):
            m = re.match(r"cam(\d+)$", os.path.basename(p))
            if m:
                cam_indices.append(int(m.group(1)))
    cam_indices = sorted(set(int(x) for x in cam_indices))

    fixed = [ci for ci in cam_indices if ci != gripper_idx]
    fixed = fixed[:3]

    if gripper_idx is None and len(cam_indices) > 0:
        # Fallback: use the highest index as gripper when metadata is missing.
        gripper_idx = cam_indices[-1]
        fixed = [ci for ci in cam_indices if ci != gripper_idx][:3]

    order = fixed.copy()
    if gripper_idx is not None:
        order.append(int(gripper_idx))

    # Fill up to 4 entries if needed.
    if len(order) < 4:
        for ci in cam_indices:
            if ci not in order:
                order.append(ci)
            if len(order) == 4:
                break

    return order[:4], gripper_idx


def load_frame_records_from_meta(meta: Dict) -> List[Dict]:
    records: List[Dict] = []
    captures = meta.get("captures", [])
    for cap in captures:
        event_id = cap.get("event_id", None)
        cams = cap.get("cams", {})
        paths = {}
        for k, v in cams.items():
            try:
                ci = int(k)
            except Exception:
                continue
            rgb_rel = v.get("rgb_path", None)
            if rgb_rel:
                paths[ci] = rgb_rel

        if event_id is None:
            # Derive from first available rgb path.
            for rgb_rel in paths.values():
                fid = parse_frame_id_from_rgb_path(rgb_rel)
                if fid is not None:
                    event_id = fid
                    break
        if event_id is None:
            continue

        records.append({"frame_id": int(event_id), "paths": paths})

    records.sort(key=lambda x: x["frame_id"])
    return records


def load_frame_records_from_files(session_root: str, cam_order: List[int]) -> List[Dict]:
    frame_to_paths: Dict[int, Dict[int, str]] = {}
    for ci in cam_order:
        pattern = os.path.join(session_root, f"cam{ci}", "rgb_*.*")
        for p in sorted(glob.glob(pattern)):
            fid = parse_frame_id_from_rgb_path(p)
            if fid is None:
                continue
            rel = os.path.relpath(p, session_root)
            frame_to_paths.setdefault(fid, {})[ci] = rel

    records = [{"frame_id": fid, "paths": frame_to_paths[fid]} for fid in sorted(frame_to_paths.keys())]
    return records


def make_blank_tile(size_hw: Tuple[int, int], text: str) -> np.ndarray:
    h, w = size_hw
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def draw_text_block(
    img: np.ndarray,
    lines: List[str],
    org_xy: Tuple[int, int] = (8, 24),
    line_h: int = 22,
    text_scale: float = 0.6,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = org_xy
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
        max_w = max(max_w, tw)
    block_h = line_h * len(lines) + 8
    cv2.rectangle(img, (x - 6, y - 20), (x + max_w + 8, y - 20 + block_h), bg_color, -1)

    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 2, cv2.LINE_AA)
        yy += line_h


def annotate_image(
    bgr: np.ndarray,
    cube: ArucoCubeTarget,
    cam_idx: int,
    is_gripper: bool,
    frame_id: int,
) -> np.ndarray:
    out = bgr.copy()

    corners, ids = cube.detect(bgr)
    n_markers = 0 if ids is None else int(len(ids))
    ids_list = [] if ids is None else [int(x) for x in ids]

    if ids is not None and len(corners) > 0:
        try:
            draw_ids = ids.reshape(-1, 1) if getattr(ids, "ndim", 1) == 1 else ids
            cv2.aruco.drawDetectedMarkers(out, corners, draw_ids)
        except Exception:
            pass

    role = "GRIPPER" if is_gripper else "FIXED"
    ids_txt = ",".join(str(x) for x in ids_list) if ids_list else "-"
    lines = [
        f"cam{cam_idx} [{role}] frame={frame_id:05d}",
        f"markers={n_markers} ids={ids_txt}",
    ]
    draw_text_block(out, lines)
    return out


def process_session(session_root: str, out_dir_name: str) -> Tuple[int, str]:
    meta = load_meta(session_root)
    cam_order, gripper_idx = choose_cam_order(meta, session_root)
    if len(cam_order) == 0:
        return 0, "no camera folders found"

    records = load_frame_records_from_meta(meta) if meta else []
    if not records:
        records = load_frame_records_from_files(session_root, cam_order)
    if not records:
        return 0, "no rgb frames found"

    # Determine tile size from first available image.
    tile_h, tile_w = None, None
    for rec in records:
        for rel in rec["paths"].values():
            p = os.path.join(session_root, rel)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                tile_h, tile_w = img.shape[:2]
                break
        if tile_h is not None:
            break
    if tile_h is None:
        return 0, "failed to load any image"

    out_dir = ensure_dir(os.path.join(session_root, out_dir_name))
    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    saved = 0
    for rec in records:
        fid = int(rec["frame_id"])
        rel_by_cam: Dict[int, str] = rec["paths"]
        tiles: List[np.ndarray] = []

        for ci in cam_order:
            rel = rel_by_cam.get(ci, f"cam{ci}/rgb_{fid:05d}.jpg")
            p = os.path.join(session_root, rel)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                tile = make_blank_tile((tile_h, tile_w), f"cam{ci} MISSING")
            else:
                if img.shape[0] != tile_h or img.shape[1] != tile_w:
                    img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                tile = annotate_image(
                    img,
                    cube=cube,
                    cam_idx=ci,
                    is_gripper=(gripper_idx is not None and ci == gripper_idx),
                    frame_id=fid,
                )
            tiles.append(tile)

        # Ensure exactly 4 tiles (2x2).
        while len(tiles) < 4:
            tiles.append(make_blank_tile((tile_h, tile_w), "EMPTY"))
        tiles = tiles[:4]

        top = cv2.hconcat([tiles[0], tiles[1]])
        bottom = cv2.hconcat([tiles[2], tiles[3]])
        quad = cv2.vconcat([top, bottom])

        out_path = os.path.join(out_dir, f"frame_{fid:05d}.jpg")
        cv2.imwrite(out_path, quad)
        saved += 1

    return saved, f"output={out_dir} cam_order={cam_order} gripper={gripper_idx}"


def main():
    parser = argparse.ArgumentParser(
        description="Build 2x2 frame-wise camera quads with marker detection overlays."
    )
    parser.add_argument("--data_root", default="./data",
                        help="Data root containing session folders (default: ./data)")
    parser.add_argument("--session_root", default=None,
                        help="Single session folder (if set, only this session is processed)")
    parser.add_argument("--out_dir_name", default="marker_quads",
                        help="Output folder name inside each session")
    args = parser.parse_args()

    if args.session_root:
        sessions = [args.session_root]
    else:
        sessions = discover_sessions(args.data_root)

    if len(sessions) == 0:
        print("[WARN] No sessions found.")
        return

    print(f"[INFO] Sessions to process: {len(sessions)}")
    total = 0
    for s in sessions:
        n, msg = process_session(s, args.out_dir_name)
        total += n
        print(f"[DONE] {s} | frames={n} | {msg}")
    print(f"[DONE] Total quad frames saved: {total}")


if __name__ == "__main__":
    main()
