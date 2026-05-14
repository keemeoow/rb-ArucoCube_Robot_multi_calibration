#!/usr/bin/env python3
"""
실제 RGB와 추정된 pose의 GLB 마스크 overlay 비교 이미지 생성.

각 객체별로:
  좌측: 실제 RGB (4-cam quad)
  우측: GLB silhouette overlay (반투명 + outline)
한 장의 side-by-side 이미지로 저장.

사용:
  python3 render_pose_overlay.py --pose_dir data/pose_fused_glb_aware
  python3 render_pose_overlay.py --pose_dir data/pose_fused
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

import sys
sys.path.insert(0, ".")
from fuse_multiframe_pose import (
    load_intrinsics, load_static_transforms, CAPTURE_DIR, DATA_DIR
)


def project_mesh_to_image(mesh, T_cam_obj, K, D, img_shape, strip_scale=True):
    """GLB → image plane silhouette.
    strip_scale=True: T_cam_obj 의 3x3에 박힌 scale 제거하고 순수 rotation만 사용
                     (ICP 학습 scale 무시, GLB 원본 크기로 렌더)
    """
    h, w = img_shape[:2]
    verts_obj = np.asarray(mesh.vertices)
    if len(verts_obj) == 0:
        return None, None
    R = T_cam_obj[:3, :3].copy()
    if strip_scale:
        # SVD로 순수 rotation 추출 (scale 제거)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
    verts_cam = (R @ verts_obj.T).T + T_cam_obj[:3, 3]
    z = verts_cam[:, 2]
    if (z > 0.05).sum() < 3:
        return None, None
    rvec = np.zeros(3); tvec = np.zeros(3)
    pts2d, _ = cv2.projectPoints(
        verts_cam.astype(np.float64), rvec, tvec,
        K.astype(np.float64), D.astype(np.float64))
    pts2d = pts2d.reshape(-1, 2)
    faces = np.asarray(mesh.faces)
    silhouette = np.zeros((h, w), dtype=np.uint8)
    for tri in faces:
        a, b, c = tri
        if z[a] <= 0.05 or z[b] <= 0.05 or z[c] <= 0.05:
            continue
        pts = np.array([pts2d[a], pts2d[b], pts2d[c]], dtype=np.int32)
        if np.any(pts[:, 0] < -200) or np.any(pts[:, 0] > w + 200):
            continue
        if np.any(pts[:, 1] < -200) or np.any(pts[:, 1] > h + 200):
            continue
        cv2.fillPoly(silhouette, [pts], 255)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROACH_NONE if hasattr(cv2, "CHAIN_APPROACH_NONE") else cv2.CHAIN_APPROX_NONE)
    return silhouette, contours


def overlay_silhouette(rgb, silhouette, contours, color_bgr, alpha=0.55,
                        outline_thickness=3):
    """반투명 채움 + 굵은 외곽선 (잘 보이게)."""
    out = rgb.copy()
    if silhouette is not None and silhouette.sum() > 0:
        color_img = np.zeros_like(out)
        color_img[:, :] = color_bgr
        mask_3 = (silhouette > 0)[..., None]
        out = np.where(mask_3, (out * (1 - alpha) + color_img * alpha).astype(np.uint8), out)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, outline_thickness, cv2.LINE_AA)
    return out


def render_for_obj(obj, T_base_obj, glb_path, intrinsics, static_T_base_cam,
                   T_gripper_cam, ref_frame, color_bgr):
    """Returns dict {cam_idx: (raw_rgb, overlay_rgb)}."""
    mesh = trimesh.load(str(glb_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    # center GLB at bbox center (same as fuse pipeline did)
    offset = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-offset)

    results = {}
    fid = f"{ref_frame:06d}"
    for ci in [0, 1, 2, 3]:
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        if ci == 2:
            T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_path.exists():
                continue
            T_be = np.load(T_be_path)
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[ci]
        T_cam_base = np.linalg.inv(T_base_cam)
        T_cam_obj = T_cam_base @ T_base_obj
        K, D, _ = intrinsics[ci]
        sil, contours = project_mesh_to_image(mesh, T_cam_obj, K, np.zeros(5), rgb.shape)
        ov = overlay_silhouette(rgb, sil, contours, color_bgr=color_bgr, alpha=0.4)
        # labels
        for img, txt in [(rgb, f"cam{ci} REAL"), (ov, f"cam{ci} {obj} POSE")]:
            cv2.rectangle(img, (5, 5), (5 + 12 * len(txt) + 10, 32), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
        results[ci] = (rgb, ov)
    return results


def stack_side_by_side(per_cam_imgs, target_w=720):
    """Overlay-only 4-cam quad."""
    ov_tiles = []
    for ci in [0, 1, 2, 3]:
        if ci not in per_cam_imgs:
            blank = np.zeros((400, target_w, 3), dtype=np.uint8)
            cv2.putText(blank, f"cam{ci}: no image", (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ov_tiles.append(blank)
            continue
        _, ov = per_cam_imgs[ci]
        h, w = ov.shape[:2]
        th = int(h * target_w / w)
        ov_tiles.append(cv2.resize(ov, (target_w, th), interpolation=cv2.INTER_AREA))
    return np.vstack([np.hstack(ov_tiles[:2]), np.hstack(ov_tiles[2:])])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose_dir", default="data/pose_fused_glb_aware")
    ap.add_argument("--objects", default="red,cream,blue,box")
    ap.add_argument("--ref_frame", type=int, default=9)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pose_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics()
    static_T_base_cam, T_gripper_cam = load_static_transforms()

    obj_colors = {
        "red":   (0, 0, 255),
        "cream": (140, 220, 240),
        "blue":  (255, 200, 100),
        "box":   (240, 240, 240),
    }
    objects = [s.strip() for s in args.objects.split(",") if s.strip()]

    # 객체별 side-by-side 이미지
    per_obj_imgs = {}
    for obj in objects:
        T_path = pose_dir / obj / "T_base_object.npy"
        glb_path = DATA_DIR / f"{obj}.glb"
        if not T_path.exists() or not glb_path.exists():
            print(f"[SKIP] {obj}: missing T or GLB")
            continue
        T = np.load(T_path)
        per_cam = render_for_obj(obj, T, glb_path, intrinsics, static_T_base_cam,
                                  T_gripper_cam, args.ref_frame,
                                  obj_colors.get(obj, (0, 255, 255)))
        side_by_side = stack_side_by_side(per_cam)
        out_path = out_dir / f"compare_{obj}.png"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"saved: {out_path}")
        per_obj_imgs[obj] = side_by_side

    # 통합 4-cam quad — 모든 객체 overlay 합쳐서
    print("\nrendering combined (all objects on each cam)...")
    fid = f"{args.ref_frame:06d}"
    combined = {}
    for ci in [0, 1, 2, 3]:
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        if ci == 2:
            T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_path.exists():
                continue
            T_be = np.load(T_be_path)
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[ci]
        T_cam_base = np.linalg.inv(T_base_cam)
        K, D, _ = intrinsics[ci]
        rgb_real = rgb.copy()
        rgb_overlay = rgb.copy()
        for obj in objects:
            T_path = pose_dir / obj / "T_base_object.npy"
            glb_path = DATA_DIR / f"{obj}.glb"
            if not T_path.exists() or not glb_path.exists():
                continue
            T_base_obj = np.load(T_path)
            mesh = trimesh.load(str(glb_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            offset = mesh.bounds.mean(axis=0)
            mesh.apply_translation(-offset)
            T_cam_obj = T_cam_base @ T_base_obj
            sil, contours = project_mesh_to_image(mesh, T_cam_obj, K, np.zeros(5), rgb.shape)
            rgb_overlay = overlay_silhouette(rgb_overlay, sil, contours,
                                              color_bgr=obj_colors.get(obj, (0, 255, 255)),
                                              alpha=0.35)
        for img, txt in [(rgb_real, f"cam{ci} REAL"), (rgb_overlay, f"cam{ci} ALL POSES")]:
            cv2.rectangle(img, (5, 5), (5 + 12 * len(txt) + 10, 32), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
        combined[ci] = (rgb_real, rgb_overlay)

    if combined:
        side_by_side = stack_side_by_side(combined)
        out_path = out_dir / "compare_all.png"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
