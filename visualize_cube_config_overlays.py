import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from aruco_cube import ArucoCubeTarget
from calibration_runtime_utils import resolve_cube_config_for_run
from config import get_default_cube_config


RAW_COLORS = [
    (0, 0, 255),
    (0, 128, 255),
    (0, 200, 255),
    (80, 255, 255),
]
CANON_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_meta(root_folder: str) -> dict:
    with open(os.path.join(root_folder, "meta.json"), "r") as f:
        return json.load(f)


def find_quad_image(quad_dir: str, event_id: int) -> str:
    path = os.path.join(quad_dir, f"frame_{int(event_id):05d}.jpg")
    return path if os.path.exists(path) else ""


def camera_tile_offsets(cam_ids: List[int], canvas_shape: Tuple[int, int, int]) -> Dict[int, Tuple[int, int]]:
    h, w = canvas_shape[:2]
    tile_w = w // 2
    tile_h = h // 2
    offsets = {}
    for idx, ci in enumerate(sorted(cam_ids)):
        row = idx // 2
        col = idx % 2
        offsets[int(ci)] = (col * tile_w, row * tile_h)
    return offsets


def draw_indexed_polygon(img: np.ndarray,
                         pts_xy: np.ndarray,
                         colors: List[Tuple[int, int, int]],
                         prefix: str,
                         radius: int = 5,
                         text_scale: float = 0.45,
                         thickness: int = 2) -> None:
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, colors[0], thickness, cv2.LINE_AA)
    for idx, pt in enumerate(pts):
        color = colors[idx % len(colors)]
        px = tuple(int(v) for v in np.round(pt))
        cv2.circle(img, px, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{prefix}{idx}",
            (px[0] + 4, px[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_event_overlay(base_img: np.ndarray,
                       cap: dict,
                       cube: ArucoCubeTarget,
                       cam_offsets: Dict[int, Tuple[int, int]]) -> np.ndarray:
    out = base_img.copy()
    for ci_str, cinfo in cap.get("cams", {}).items():
        ci = int(ci_str)
        if ci not in cam_offsets:
            continue
        ox, oy = cam_offsets[ci]
        for marker in cinfo.get("markers", []):
            mid = int(marker["marker_id"])
            raw = np.asarray(marker["corners_2d"], dtype=np.float64).reshape(4, 2)
            canon = cube.model.reorder_image_corners(mid, raw)
            raw_shift = raw + np.array([ox, oy], dtype=np.float64)
            canon_shift = canon + np.array([ox, oy], dtype=np.float64)
            draw_indexed_polygon(out, raw_shift, RAW_COLORS, prefix="r", radius=4, text_scale=0.40, thickness=1)
            draw_indexed_polygon(out, canon_shift, CANON_COLORS, prefix="c", radius=3, text_scale=0.40, thickness=2)
            anchor = tuple(int(v) for v in np.round(np.mean(raw_shift, axis=0)))
            reorder = cube.cfg.corner_reorder.get(mid, [0, 1, 2, 3])
            face = cube.model.marker_face_name(mid)
            roll = float(cube.cfg.face_roll_deg.get(mid, 0.0))
            explicit = cube.model.uses_explicit_marker_pose(mid)
            label = f"id{mid} {face} perm={reorder} roll={roll:.0f}{' explicit' if explicit else ''}"
            cv2.putText(out, label, (anchor[0] - 30, anchor[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, label, (anchor[0] - 30, anchor[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def marker_pose_summary(cube: ArucoCubeTarget) -> List[dict]:
    rows = []
    for mid in sorted(cube.cfg.marker_ids):
        T = cube.model.marker_pose_in_rig(mid)
        rows.append({
            "marker_id": int(mid),
            "face": cube.model.marker_face_name(mid),
            "corner_reorder": list(cube.cfg.corner_reorder.get(mid, [0, 1, 2, 3])),
            "face_roll_deg": float(cube.cfg.face_roll_deg.get(mid, 0.0)),
            "uses_explicit_marker_pose": bool(cube.model.uses_explicit_marker_pose(mid)),
            "marker_pose_4x4": np.asarray(T, dtype=float).tolist(),
        })
    return rows


def draw_canonical_square(ax, reorder: List[int], title: str) -> None:
    raw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    canon = raw[np.asarray(reorder, dtype=int)]
    ax.plot(*np.vstack([raw, raw[0]]).T, color="0.7", linewidth=1.0)
    ax.plot(*np.vstack([canon, canon[0]]).T, color="tab:green", linewidth=2.0)
    for i, pt in enumerate(raw):
        ax.scatter(pt[0], pt[1], c="tab:red", s=30)
        ax.text(pt[0] - 0.08, pt[1] - 0.06, f"r{i}", color="tab:red", fontsize=8)
    for i, pt in enumerate(canon):
        ax.scatter(pt[0], pt[1], c="tab:green", s=18)
        ax.text(pt[0] + 0.03, pt[1] + 0.03, f"c{i}", color="tab:green", fontsize=8)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def save_config_summary_figure(cube: ArucoCubeTarget, out_path: str) -> None:
    mids = sorted(cube.cfg.marker_ids)
    fig, axes = plt.subplots(len(mids), 2, figsize=(12, 2.5 * len(mids)))
    if len(mids) == 1:
        axes = np.asarray([axes])
    for row_idx, mid in enumerate(mids):
        left = axes[row_idx, 0]
        right = axes[row_idx, 1]
        reorder = list(cube.cfg.corner_reorder.get(mid, [0, 1, 2, 3]))
        face = cube.model.marker_face_name(mid)
        roll = float(cube.cfg.face_roll_deg.get(mid, 0.0))
        explicit = cube.model.uses_explicit_marker_pose(mid)
        draw_canonical_square(left, reorder, f"id{mid} {face}")

        T = cube.model.marker_pose_in_rig(mid)
        right.axis("off")
        text = [
            f"id: {mid}",
            f"face: {face}",
            f"corner_reorder: {reorder}",
            f"face_roll_deg: {roll:.1f}",
            f"marker_pose_4x4 explicit: {explicit}",
            "",
            "marker_pose_in_rig:",
        ]
        for r in T:
            text.append("  " + " ".join(f"{v: .3f}" for v in r))
        right.text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=9)
    fig.suptitle("Cube Config Summary: corner_reorder / face_roll_deg / marker_pose_4x4", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cube_faces_vertices(half_side: float) -> Dict[str, np.ndarray]:
    d = float(half_side)
    return {
        "+Z": np.array([[-d, -d, d], [d, -d, d], [d, d, d], [-d, d, d]], dtype=float),
        "-Z": np.array([[d, -d, -d], [-d, -d, -d], [-d, d, -d], [d, d, -d]], dtype=float),
        "+X": np.array([[d, -d, d], [d, -d, -d], [d, d, -d], [d, d, d]], dtype=float),
        "-X": np.array([[-d, -d, -d], [-d, -d, d], [-d, d, d], [-d, d, -d]], dtype=float),
        "+Y": np.array([[-d, d, d], [d, d, d], [d, d, -d], [-d, d, -d]], dtype=float),
        "-Y": np.array([[-d, -d, -d], [d, -d, -d], [d, -d, d], [-d, -d, d]], dtype=float),
    }


def save_cube_3d_figure(cube: ArucoCubeTarget, out_path: str) -> None:
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    half = cube.cfg.cube_side_m / 2.0
    faces = cube_faces_vertices(half)
    face_poly = Poly3DCollection(list(faces.values()), facecolors="#d9d9d9", edgecolors="0.5", alpha=0.18)
    ax.add_collection3d(face_poly)

    axis_scale = cube.cfg.marker_size_m * 0.5
    for mid in sorted(cube.cfg.marker_ids):
        T = cube.model.marker_pose_in_rig(mid)
        corners = cube.model.marker_corners_in_rig(mid)
        poly = Poly3DCollection([corners], alpha=0.35, facecolors="tab:blue", edgecolors="k")
        ax.add_collection3d(poly)
        for idx, pt in enumerate(corners):
            ax.scatter(pt[0], pt[1], pt[2], c="k", s=15)
            ax.text(pt[0], pt[1], pt[2], f"{mid}:{idx}", fontsize=8)
        origin = T[:3, 3]
        u = T[:3, 0] * axis_scale
        v = T[:3, 1] * axis_scale
        n = T[:3, 2] * axis_scale
        ax.quiver(origin[0], origin[1], origin[2], u[0], u[1], u[2], color="r", linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2], color="g", linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2], n[0], n[1], n[2], color="b", linewidth=2)
        label = f"id{mid} {cube.model.marker_face_name(mid)}"
        if cube.model.uses_explicit_marker_pose(mid):
            label += " explicit"
        ax.text(origin[0], origin[1], origin[2], label, fontsize=9)

    lim = cube.cfg.cube_side_m * 0.9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cube Marker Poses in Object Frame\nred=u, green=v, blue=normal")
    ax.view_init(elev=24, azim=38)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cube config over marker_quads and 3D cube model")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--calib_dir", default=None)
    parser.add_argument("--cube_config_json", default=None)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    root = os.path.abspath(args.root_folder)
    out_dir = ensure_dir(args.out_dir or os.path.join(root, "config_viz"))
    quad_out_dir = ensure_dir(os.path.join(out_dir, "marker_quads_overlay"))
    meta = load_meta(root)

    cube_cfg, cube_cfg_source = resolve_cube_config_for_run(
        root_folder=root,
        calib_dir=args.calib_dir,
        cube_config_json=args.cube_config_json,
        default_cfg=get_default_cube_config(),
    )
    cube = ArucoCubeTarget(cube_cfg)

    summary_rows = marker_pose_summary(cube)
    with open(os.path.join(out_dir, "cube_config_pose_summary.json"), "w") as f:
        json.dump({
            "cube_config_source": cube_cfg_source,
            "marker_rows": summary_rows,
        }, f, indent=2)

    save_config_summary_figure(cube, os.path.join(out_dir, "cube_config_summary.png"))
    save_cube_3d_figure(cube, os.path.join(out_dir, "cube_marker_poses_3d.png"))

    quad_dir = os.path.join(root, "marker_quads")
    written = 0
    for cap in meta.get("captures", []):
        event_id = int(cap.get("event_id", -1))
        quad_path = find_quad_image(quad_dir, event_id)
        if not quad_path:
            continue
        img = cv2.imread(quad_path)
        if img is None:
            continue
        cam_ids = [int(k) for k, v in cap.get("cams", {}).items() if v.get("saved")]
        offsets = camera_tile_offsets(cam_ids, img.shape)
        overlay = draw_event_overlay(img, cap, cube, offsets)
        header = f"cube_cfg={cube_cfg_source} | raw=r* | canonical=c*"
        cv2.putText(overlay, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
        out_path = os.path.join(quad_out_dir, f"frame_{event_id:05d}_overlay.jpg")
        cv2.imwrite(out_path, overlay)
        written += 1
        if args.max_frames > 0 and written >= int(args.max_frames):
            break

    print(f"[SAVE] {os.path.join(out_dir, 'cube_config_summary.png')}")
    print(f"[SAVE] {os.path.join(out_dir, 'cube_marker_poses_3d.png')}")
    print(f"[SAVE] {os.path.join(out_dir, 'cube_config_pose_summary.json')}")
    print(f"[SAVE] {quad_out_dir} ({written} frame overlays)")


if __name__ == "__main__":
    main()
