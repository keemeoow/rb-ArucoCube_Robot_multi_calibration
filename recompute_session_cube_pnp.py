import argparse
import json
import os
import shutil
from typing import Dict

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, depth_metrics_to_fields, rodrigues_to_Rt
from calibration_runtime_utils import load_intrinsics_with_depth_scale, resolve_cube_config_for_run
from capture_detection_utils import detect_cube_markers_in_frame
from charuco_utils import CharucoTarget
from config import CharucoBoardConfig, CubeConfig, get_default_cube_config
from cube_config_utils import cube_config_to_dict


def main():
    ap = argparse.ArgumentParser(description="Recompute cube_pnp from saved session images")
    ap.add_argument("--root_folder", required=True)
    ap.add_argument("--intrinsics_dir", required=True)
    ap.add_argument("--cube_config_json", type=str, default=None,
                    help="Optional cube config JSON override. Leave unset to use the project's canonical cube definition.")
    ap.add_argument("--board_mask_pad_px", type=float, default=6.0)
    ap.add_argument("--gripper_cube_min_markers", type=int, default=1)
    ap.add_argument("--gripper_cube_min_aspect", type=float, default=0.35)
    ap.add_argument("--solve_reproj_thr", type=float, default=10.0)
    ap.add_argument("--write_inplace", action="store_true")
    ap.add_argument("--out_meta", type=str, default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.root_folder)
    meta_path = os.path.join(root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    gripper_cam_idx = int(meta.get("gripper_cam_idx"))
    cube_cfg, cube_cfg_source = resolve_cube_config_for_run(
        root_folder=root,
        cube_config_json=args.cube_config_json,
        default_cfg=get_default_cube_config(),
    )
    cube = ArucoCubeTarget(cube_cfg)
    charuco = CharucoTarget(CharucoBoardConfig())
    print(f"[INFO] Cube config source: {cube_cfg_source}")
    meta["cube_config"] = cube_config_to_dict(cube_cfg)
    meta["cube_config_source"] = cube_cfg_source

    cam_intr: Dict[int, tuple] = {}
    for ci_s in meta.get("cam_indices", []):
        ci = int(ci_s)
        cam_intr[ci] = load_intrinsics_with_depth_scale(args.intrinsics_dir, ci)

    summary = {
        "root_folder": root,
        "gripper_cam_idx": gripper_cam_idx,
        "cube_config_source": cube_cfg_source,
        "cube_config_used": cube_config_to_dict(cube_cfg),
        "board_mask_pad_px": float(args.board_mask_pad_px),
        "captures": [],
    }

    for cap in meta.get("captures", []):
        event_id = int(cap.get("event_id", -1))
        cap_summary = {
            "event_id": event_id,
            "set_index": cap.get("set_index"),
            "cams": {},
        }
        for ci_s, cinfo in cap.get("cams", {}).items():
            ci = int(ci_s)
            rgb_rel = cinfo.get("rgb_path")
            if not rgb_rel:
                continue
            rgb_path = os.path.join(root, rgb_rel)
            img = cv2.imread(rgb_path)
            if img is None:
                cap_summary["cams"][ci_s] = {"status": "missing_rgb"}
                continue

            depth = None
            depth_rel = cinfo.get("depth_path")
            if depth_rel:
                depth = cv2.imread(os.path.join(root, depth_rel), cv2.IMREAD_UNCHANGED)

            K, D, depth_scale = cam_intr[ci]
            detect_info = detect_cube_markers_in_frame(
                img,
                cube,
                cube_ids=cube_cfg.marker_ids,
                charuco=charuco if ci == gripper_cam_idx else None,
                is_gripper=(ci == gripper_cam_idx),
                board_mask_pad_px=float(args.board_mask_pad_px),
            )
            ids = detect_info["ids"]
            marker_ids = [] if ids is None else [int(x) for x in np.asarray(ids).reshape(-1)]
            n_markers = len(marker_ids)

            min_markers = int(args.gripper_cube_min_markers) if ci == gripper_cam_idx else 1
            min_aspect = float(args.gripper_cube_min_aspect) if ci == gripper_cam_idx else 0.0
            ok, rvec, tvec, used_ids, reproj = cube.solve_pnp_cube(
                detect_info["cube_image"], K, D,
                use_ransac=True,
                min_markers=max(min_markers, 1),
                reproj_thr_mean_px=float(args.solve_reproj_thr),
                return_reproj=True,
                min_aspect=min_aspect,
                depth_u16=depth,
                depth_scale=depth_scale,
            )

            cinfo["n_markers_detected"] = int(n_markers)
            cinfo["marker_ids"] = marker_ids
            cinfo["cube_detect_raw_ids"] = list(detect_info["raw_ids"])
            cinfo["cube_detect_filtered_ids"] = list(detect_info["filtered_ids"])
            cinfo["board_mask_applied"] = bool(detect_info["board_mask_applied"])
            cinfo["cube_visible"] = bool(n_markers >= 1)
            if ci == gripper_cam_idx:
                cinfo["charuco_detect_n"] = int(detect_info["charuco_detect_n"])

            if ok and reproj is not None:
                T_cam_cube = rodrigues_to_Rt(rvec, tvec)
                cinfo["cube_pnp"] = {
                    "ok": True,
                    "rvec": rvec.flatten().tolist(),
                    "tvec": tvec.flatten().tolist(),
                    "used_ids": [int(x) for x in used_ids],
                    "reproj_mean_px": float(reproj["err_mean"]),
                    "n_points": int(reproj["n_points"]),
                    "T_cam_cube_4x4": T_cam_cube.tolist(),
                    "min_markers_required": int(min_markers),
                    "min_aspect_required": float(min_aspect),
                    **depth_metrics_to_fields(reproj.get("depth_metrics")),
                }
            else:
                cinfo.pop("cube_pnp", None)

            cap_summary["cams"][ci_s] = {
                "status": "ok" if ok else "failed",
                "marker_ids": marker_ids,
                "raw_ids": list(detect_info["raw_ids"]),
                "board_mask_applied": bool(detect_info["board_mask_applied"]),
                "charuco_detect_n": int(detect_info["charuco_detect_n"]) if ci == gripper_cam_idx else None,
                "used_ids": [int(x) for x in used_ids] if used_ids else [],
                "reproj_mean_px": None if reproj is None else float(reproj["err_mean"]),
                "depth_valid": None if reproj is None else bool((reproj.get("depth_metrics") or {}).get("valid", False)),
                "depth_plane_mean_mm": None if reproj is None else (reproj.get("depth_metrics") or {}).get("plane_mean_mm"),
            }
        summary["captures"].append(cap_summary)

    out_meta = args.out_meta
    if args.write_inplace:
        backup_path = os.path.join(root, "meta.before_boardmask_cube_pnp_recompute.json")
        if not os.path.exists(backup_path):
            shutil.copy2(meta_path, backup_path)
        out_meta = meta_path
    if not out_meta:
        out_meta = os.path.join(root, "meta_recomputed_cube_pnp.json")

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    summary_path = os.path.join(root, "gripper_cube_pnp_recompute_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] {out_meta}")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
