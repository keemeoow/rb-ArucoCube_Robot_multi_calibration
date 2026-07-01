# Step1b_charuco_intrinsics.py

"""
Step 1b: ChArUco 보드로 카메라 color intrinsics (K, D) 재캘리브레이션.

Step1은 RealSense 공장(factory) intrinsics를 그대로 덤프한다. 그런데 D415/D435
color 스트림의 공장 왜곡계수(D)는 전부 0으로 보고되어 실제 렌즈 왜곡이 보정되지
않는다. 이 스크립트는 ChArUco 보드를 라이브 대화형으로 촬영해 각 카메라의 color
K, D를 직접 추정하고, 기존 intrinsics/cam{idx}.npz의 color_K/color_D만 교체한다.
depth 관련 필드(depth_K, depth_scale, R_depth_to_color 등)와 해상도/시리얼은 그대로
보존하므로 다운스트림(Step2~5) 코드는 수정할 필요가 없다.

전제조건:
  - 먼저 Step1_dump_all_intrinsics.py 를 실행해 device_map.json 과 cam{idx}.npz
    (depth 필드 포함)를 만들어 두어야 한다.
  - config.py 의 CharucoBoardConfig 가 실제 인쇄된 보드와 일치해야 한다
    (기본: 11x7, square 25mm, marker 18mm, DICT_4X4_250, marker_id_start=5).
    * intrinsics(K, D)는 보드의 물리 치수와 무관하지만, 보드 레이아웃(칸 수,
      dictionary, 마커 ID)은 검출을 위해 반드시 일치해야 한다.

동작:
  1) device_map.json 로드 (serial -> cam_idx, gripper_cam_idx)
  2) 연결된 각 카메라를 하나씩 열고, ChArUco 보드를 흔들며 다양한 각도/위치에서
     프레임을 수집 (자동 그랩 + 수동 SPACE). 화면에 검출/커버리지/선명도 피드백.
  3) board.matchImagePoints + cv2.calibrateCamera 로 2-pass(이상치 제거) 보정
  4) intrinsics/cam{idx}.npz 의 color_K/color_D 교체 (factory 값은 백업 보존)
  5) intrinsics/charuco_intrinsics_report.json 리포트 저장

키 조작 (카메라별 수집 중):
  SPACE : 현재 프레임 수동 그랩
  a     : 자동 그랩 on/off
  u     : 마지막 그랩 취소(undo)
  c/Enter: 이 카메라 수집 종료 -> 보정 실행
  s     : 이 카메라 건너뛰기 (factory 값 유지)
  q     : 전체 중단 (아무것도 쓰지 않음)

명령어 예시:
  python Step1b_charuco_intrinsics.py --intr_dir ./intrinsics --target_views 30
  python Step1b_charuco_intrinsics.py --rational           # 8-계수 rational 모델
  python Step1b_charuco_intrinsics.py --no-auto_grab       # 수동 그랩만
"""

import os
import json
import time
import argparse

import numpy as np
import cv2

from camera import RealSenseCamera
from charuco_utils import CharucoTarget
from config import CharucoBoardConfig


# ---------------------------------------------------------------------------
# 보정 헬퍼
# ---------------------------------------------------------------------------
def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance = 선명도 지표 (높을수록 선명)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _obj_img_from_charuco(board, ch_corners, ch_ids):
    """ChArUco 코너/ID -> (objectPoints Nx1x3 f32, imagePoints Nx1x2 f32).

    OpenCV 4.7+ 의 board.matchImagePoints 를 우선 사용 (이 환경 4.13 지원 확인됨).
    구버전 대비 chessboardCorners 인덱싱 폴백도 둔다.
    """
    if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
        return None, None

    if hasattr(board, "matchImagePoints"):
        try:
            obj, img = board.matchImagePoints(ch_corners, ch_ids)
            if obj is not None and img is not None and len(obj) >= 4:
                return obj.reshape(-1, 1, 3).astype(np.float32), img.reshape(-1, 1, 2).astype(np.float32)
        except Exception:
            pass

    # 폴백: chessboardCorners 를 ID로 인덱싱
    if hasattr(board, "getChessboardCorners"):
        chess = np.asarray(board.getChessboardCorners())
    elif hasattr(board, "chessboardCorners"):
        chess = np.asarray(board.chessboardCorners)
    else:
        return None, None
    chess = chess.reshape(-1, 3)
    ids = np.asarray(ch_ids).reshape(-1)
    if ids.max() >= len(chess):
        return None, None
    obj = chess[ids].reshape(-1, 1, 3).astype(np.float32)
    img = np.asarray(ch_corners).reshape(-1, 1, 2).astype(np.float32)
    return obj, img


def _run_calib(views, image_size, flags, K0=None, D0=None):
    """views: [(obj, img), ...] -> (rms, K, D, per_view_rms)."""
    obj_list = [v[0] for v in views]
    img_list = [v[1] for v in views]

    if K0 is not None:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, image_size,
            K0.copy(), (None if D0 is None else D0.copy()),
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
        )
    else:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, image_size, None, None, flags=flags
        )

    per_view = []
    for i, (o, im) in enumerate(views):
        proj, _ = cv2.projectPoints(o, rvecs[i], tvecs[i], K, D)
        diff = proj.reshape(-1, 2) - im.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))
    return float(rms), K, D, per_view


def calibrate_intrinsics(board, accepted, image_size, flags, K0=None, D0=None):
    """2-pass 보정: 1차 보정 -> per-view 이상치 제거 -> 2차 보정.

    accepted: [(ch_corners, ch_ids), ...]
    반환: dict 또는 None (뷰 부족)
    """
    views = []
    for ch_c, ch_id in accepted:
        obj, img = _obj_img_from_charuco(board, ch_c, ch_id)
        if obj is not None:
            views.append((obj, img))

    if len(views) < 4:
        return None

    rms, K, D, per = _run_calib(views, image_size, flags, K0, D0)
    per_arr = np.asarray(per)
    thr = float(per_arr.mean() + per_arr.std())

    keep = [v for v, e in zip(views, per) if e <= thr]
    dropped = len(views) - len(keep)

    if len(keep) >= 4 and dropped > 0:
        rms2, K2, D2, per2 = _run_calib(keep, image_size, flags, K0, D0)
        return {
            "rms": rms2, "K": K2, "D": D2,
            "n_used": len(keep), "n_total": len(views),
            "n_dropped": dropped, "reject_thr_px": thr,
            "per_view": per2,
        }

    return {
        "rms": rms, "K": K, "D": D,
        "n_used": len(views), "n_total": len(views),
        "n_dropped": 0, "reject_thr_px": thr,
        "per_view": per,
    }


# ---------------------------------------------------------------------------
# 라이브 수집 (카메라 1대)
# ---------------------------------------------------------------------------
def _draw_overlay(vis, coverage, n_accepted, target, sharp, blur_thr,
                  auto_on, n_corners, cov_cols, cov_rows):
    h, w = vis.shape[:2]

    # 커버리지 그리드
    for r in range(cov_rows):
        for c in range(cov_cols):
            x0 = int(c * w / cov_cols)
            y0 = int(r * h / cov_rows)
            x1 = int((c + 1) * w / cov_cols)
            y1 = int((r + 1) * h / cov_rows)
            hit = coverage[r, c] > 0
            col = (0, 180, 0) if hit else (60, 60, 60)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)

    covered = int((coverage > 0).sum())
    total_cells = cov_cols * cov_rows

    sharp_ok = sharp >= blur_thr
    lines = [
        f"cam views: {n_accepted}/{target}   coverage: {covered}/{total_cells} cells",
        f"charuco corners: {n_corners}   sharp: {sharp:.0f} ({'OK' if sharp_ok else 'BLUR'})",
        f"auto-grab: {'ON' if auto_on else 'OFF'}   [SPACE]grab [a]auto [u]undo [c]done [s]skip [q]quit",
    ]
    y = 22
    for i, t in enumerate(lines):
        color = (255, 255, 255)
        if i == 1 and not sharp_ok:
            color = (0, 165, 255)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22
    return vis


def collect_for_camera(cam, target, cam_idx, is_gripper, args, save_dir):
    """한 카메라에서 프레임 수집. 반환: (status, accepted, image_size)
    status: 'done' | 'skip' | 'abort'
    """
    win = f"cam{cam_idx} charuco intrinsics ({'GRIPPER' if is_gripper else 'FIXED'})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    accepted = []          # [(ch_corners, ch_ids), ...]
    coverage = np.zeros((args.cov_rows, args.cov_cols), dtype=int)
    last_centroid = None
    last_grab_t = 0.0
    auto_on = bool(args.auto_grab)
    image_size = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n[cam{cam_idx}] 보드를 카메라 앞에서 다양한 각도/거리/위치로 움직이세요. "
          f"목표 {args.target_views}장. (c=보정, s=건너뛰기)")

    while True:
        color, _, _ = cam.get_latest()
        if color is None:
            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                cv2.destroyWindow(win)
                return "abort", accepted, image_size
            continue

        h, w = color.shape[:2]
        image_size = (w, h)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        sharp = _sharpness(gray)

        ch_c, ch_id, n_corners, m_c, m_id = target.detect(color)

        vis = color.copy()
        if m_c is not None and len(m_c) > 0:
            cv2.aruco.drawDetectedMarkers(vis, m_c, m_id)
        if ch_c is not None and ch_id is not None and n_corners > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_id, (0, 255, 255))

        # 자동 그랩 판정
        do_grab = False
        centroid = None
        if ch_c is not None and n_corners >= args.min_corners:
            centroid = ch_c.reshape(-1, 2).mean(axis=0)
            moved = (last_centroid is None) or \
                    (float(np.linalg.norm(centroid - last_centroid)) > args.move_thresh)
            now = time.time()
            if (auto_on and sharp >= args.blur_thresh and moved
                    and (now - last_grab_t) > args.grab_cooldown):
                do_grab = True

        _draw_overlay(vis, coverage, len(accepted), args.target_views, sharp,
                      args.blur_thresh, auto_on, n_corners, args.cov_cols, args.cov_rows)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(win)
            return "abort", accepted, image_size
        elif key == ord('s'):
            cv2.destroyWindow(win)
            return "skip", accepted, image_size
        elif key in (ord('c'), 13, 10):
            cv2.destroyWindow(win)
            return "done", accepted, image_size
        elif key == ord('a'):
            auto_on = not auto_on
        elif key == ord('u'):
            if accepted:
                accepted.pop()
                # 커버리지는 근사치 유지 (정확 복원 대신 재계산)
                coverage[:] = 0
                for cc, _ in accepted:
                    for px, py in cc.reshape(-1, 2):
                        cc_col = min(args.cov_cols - 1, int(px * args.cov_cols / w))
                        cc_row = min(args.cov_rows - 1, int(py * args.cov_rows / h))
                        coverage[cc_row, cc_col] += 1
                last_centroid = None
                print(f"[cam{cam_idx}] undo -> {len(accepted)} views")
        elif key == ord(' '):
            if ch_c is not None and n_corners >= args.min_corners:
                do_grab = True
                if centroid is None:
                    centroid = ch_c.reshape(-1, 2).mean(axis=0)
            else:
                print(f"[cam{cam_idx}] 그랩 불가: charuco 코너 {n_corners} < "
                      f"{args.min_corners} (보드를 더 잘 보이게)")

        if do_grab:
            accepted.append((ch_c, ch_id))
            for px, py in ch_c.reshape(-1, 2):
                cc_col = min(args.cov_cols - 1, int(px * args.cov_cols / w))
                cc_row = min(args.cov_rows - 1, int(py * args.cov_rows / h))
                coverage[cc_row, cc_col] += 1
            last_centroid = centroid
            last_grab_t = time.time()
            if save_dir:
                fn = os.path.join(save_dir, f"view_{len(accepted):03d}.png")
                cv2.imwrite(fn, color)
            print(f"[cam{cam_idx}] grab #{len(accepted)}  corners={n_corners}  sharp={sharp:.0f}")

            if len(accepted) >= args.target_views:
                print(f"[cam{cam_idx}] 목표 {args.target_views}장 도달. c/Enter로 보정하거나 계속 수집.")


# ---------------------------------------------------------------------------
# npz 갱신 / 리포트
# ---------------------------------------------------------------------------
def overwrite_color_intrinsics(npz_path, backup_dir, K, D, result, serial):
    """cam{idx}.npz 의 color_K/color_D 만 교체하고 나머지 필드는 보존.
    최초 1회에 한해 원본(factory) 전체를 backup_dir 에 복사하고,
    npz 안에도 factory_color_K/D 를 남긴다.
    """
    d = dict(np.load(npz_path, allow_pickle=True))

    # 파일 단위 factory 백업 (최초 1회만 — 재실행해도 진짜 factory 보존)
    os.makedirs(backup_dir, exist_ok=True)
    base = os.path.basename(npz_path)
    backup_path = os.path.join(backup_dir, base)
    if not os.path.exists(backup_path):
        np.savez(backup_path, **d)

    # npz 내부에도 최초 factory 값 보존
    if "factory_color_K" not in d:
        d["factory_color_K"] = np.asarray(d["color_K"], dtype=np.float64)
        d["factory_color_D"] = np.asarray(d["color_D"], dtype=np.float64)

    d["color_K"] = np.asarray(K, dtype=np.float64)
    d["color_D"] = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    d["intrinsics_source"] = "charuco"
    d["charuco_reproj_error_px"] = float(result["rms"])
    d["charuco_num_views"] = int(result["n_used"])
    d["charuco_calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    np.savez(npz_path, **d)
    return backup_path


def main():
    parser = argparse.ArgumentParser(
        description="ChArUco 기반 color intrinsics 재캘리브레이션 (Step1 결과에 덮어쓰기)")
    parser.add_argument("--intr_dir", type=str, default="intrinsics")
    parser.add_argument("--target_views", type=int, default=30,
                        help="카메라당 목표 수집 장수")
    parser.add_argument("--min_views", type=int, default=12,
                        help="보정에 필요한 최소 수집 장수 (미만이면 건너뜀)")
    parser.add_argument("--min_corners", type=int, default=8,
                        help="한 프레임을 그랩하기 위한 최소 charuco 코너 수")
    parser.add_argument("--blur_thresh", type=float, default=60.0,
                        help="Laplacian variance 최소값 (자동 그랩 선명도 기준)")
    parser.add_argument("--move_thresh", type=float, default=25.0,
                        help="자동 그랩 시 직전 그랩 대비 코너 중심 이동 최소 px")
    parser.add_argument("--grab_cooldown", type=float, default=0.6,
                        help="자동 그랩 최소 간격 (초)")
    parser.add_argument("--cov_cols", type=int, default=4)
    parser.add_argument("--cov_rows", type=int, default=3)
    parser.add_argument("--auto_grab", action=argparse.BooleanOptionalAction, default=True,
                        help="자동 그랩 사용 (--no-auto_grab 으로 수동만)")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True,
                        help="수집 프레임을 intr_dir/charuco_capture/cam{idx}/ 에 저장")
    parser.add_argument("--reset_devices", action=argparse.BooleanOptionalAction, default=True,
                        help="시작 시 모든 RealSense 하드웨어 리셋")
    parser.add_argument("--rational", action="store_true",
                        help="8-계수 CALIB_RATIONAL_MODEL 사용 (기본: 5-계수 Brown-Conrady)")
    parser.add_argument("--fix_aspect", action="store_true",
                        help="fx/fy 비율 고정 (CALIB_FIX_ASPECT_RATIO)")
    parser.add_argument("--use_factory_guess", action="store_true",
                        help="factory K를 초기 추정값으로 사용 (수렴 안정화)")
    # 보드 설정 override (기본은 config.py CharucoBoardConfig)
    parser.add_argument("--squares_x", type=int, default=None)
    parser.add_argument("--squares_y", type=int, default=None)
    parser.add_argument("--square_len_m", type=float, default=None)
    parser.add_argument("--marker_len_m", type=float, default=None)
    parser.add_argument("--dictionary", type=str, default=None)
    parser.add_argument("--marker_id_start", type=int, default=None)
    args = parser.parse_args()

    intr_dir = args.intr_dir
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        print(f"[ERROR] {map_path} 없음. 먼저 Step1_dump_all_intrinsics.py 를 실행하세요.")
        return
    with open(map_path, "r") as f:
        dev_map = json.load(f)
    serial_to_idx = dev_map.get("serial_to_idx", {})
    gripper_cam_idx = dev_map.get("gripper_cam_idx", None)
    if not serial_to_idx:
        print("[ERROR] device_map.json 에 serial_to_idx 가 비어있음.")
        return

    # 보드 설정 구성
    cfg = CharucoBoardConfig()
    if args.squares_x is not None:
        cfg.squares_x = args.squares_x
    if args.squares_y is not None:
        cfg.squares_y = args.squares_y
    if args.square_len_m is not None:
        cfg.square_length_m = args.square_len_m
    if args.marker_len_m is not None:
        cfg.marker_length_m = args.marker_len_m
    if args.dictionary is not None:
        cfg.dictionary_name = args.dictionary
    if args.marker_id_start is not None:
        cfg.marker_id_start = args.marker_id_start
    target = CharucoTarget(cfg)
    print(f"[INFO] ChArUco board: {cfg.squares_x}x{cfg.squares_y}  "
          f"square={cfg.square_length_m*1000:.0f}mm marker={cfg.marker_length_m*1000:.0f}mm  "
          f"dict={cfg.dictionary_name} id_start={cfg.marker_id_start}")

    # 보정 flags
    flags = 0
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    print(f"[INFO] dist model: {'RATIONAL(8)' if args.rational else 'BROWN-CONRADY(5)'}  "
          f"OpenCV {cv2.__version__}")

    if args.reset_devices:
        RealSenseCamera.reset_all_devices()

    connected = RealSenseCamera.list_devices()  # {serial: name}
    idx_pairs = sorted(
        [(int(serial_to_idx[s]), s) for s in connected if s in serial_to_idx],
        key=lambda x: x[0],
    )
    if not idx_pairs:
        print("[ERROR] device_map 에 매핑된 연결 카메라가 없음.")
        return
    print(f"[INFO] 재보정 대상 {len(idx_pairs)}대: "
          + ", ".join(f"cam{i}({'GRIP' if i == gripper_cam_idx else 'FIX'})" for i, _ in idx_pairs))

    backup_dir = os.path.join(intr_dir, "factory_backup")
    report = {
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "opencv": cv2.__version__,
        "dist_model": "rational8" if args.rational else "brown_conrady5",
        "board": {
            "squares_x": cfg.squares_x, "squares_y": cfg.squares_y,
            "square_length_m": cfg.square_length_m, "marker_length_m": cfg.marker_length_m,
            "dictionary": cfg.dictionary_name, "marker_id_start": cfg.marker_id_start,
        },
        "cameras": {},
    }

    aborted = False
    for cam_idx, serial in idx_pairs:
        npz_path = os.path.join(intr_dir, f"cam{cam_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] {npz_path} 없음 -> cam{cam_idx} 건너뜀 (Step1 먼저 실행).")
            report["cameras"][str(cam_idx)] = {"serial": serial, "status": "no_npz"}
            continue

        d0 = np.load(npz_path, allow_pickle=True)
        w = int(d0["color_w"]); h = int(d0["color_h"]); fps = int(d0["fps"])
        factory_K = np.asarray(d0["color_K"], dtype=np.float64)
        is_gripper = (cam_idx == gripper_cam_idx)

        print(f"\n{'='*64}\n[cam{cam_idx}] serial={serial}  {w}x{h}@{fps}  "
              f"{'GRIPPER' if is_gripper else 'FIXED'}\n{'='*64}")

        cam = RealSenseCamera(serial, width=w, height=h, fps=fps,
                              use_color=True, use_depth=False)
        try:
            cam.start()
        except Exception as e:
            print(f"[WARN] cam{cam_idx} 시작 실패: {e} -> 건너뜀")
            report["cameras"][str(cam_idx)] = {"serial": serial, "status": "start_failed"}
            continue

        save_dir = (os.path.join(intr_dir, "charuco_capture", f"cam{cam_idx}")
                    if args.save_images else None)
        try:
            status, accepted, image_size = collect_for_camera(
                cam, target, cam_idx, is_gripper, args, save_dir)
        finally:
            cam.stop()

        if status == "abort":
            print("[INFO] 사용자 중단(q). 지금까지 쓴 것 외에는 변경 없음.")
            aborted = True
            break
        if status == "skip":
            print(f"[cam{cam_idx}] 건너뜀 -> factory 값 유지.")
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": "skipped", "num_views": len(accepted)}
            continue

        if len(accepted) < args.min_views:
            print(f"[cam{cam_idx}] 수집 {len(accepted)} < min_views {args.min_views} "
                  f"-> 보정 생략, factory 유지.")
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": "too_few_views", "num_views": len(accepted)}
            continue

        if image_size is None:
            image_size = (w, h)

        K0 = factory_K if args.use_factory_guess else None
        result = calibrate_intrinsics(target.board, accepted, image_size, flags, K0=K0)
        if result is None:
            print(f"[cam{cam_idx}] 보정 실패 (유효 뷰 부족).")
            report["cameras"][str(cam_idx)] = {
                "serial": serial, "status": "calib_failed", "num_views": len(accepted)}
            continue

        K, D = result["K"], result["D"]
        print(f"[cam{cam_idx}] RMS reproj = {result['rms']:.4f} px  "
              f"(used {result['n_used']}/{result['n_total']}, dropped {result['n_dropped']})")
        print(f"           factory fx,fy,cx,cy = "
              f"{factory_K[0,0]:.2f},{factory_K[1,1]:.2f},{factory_K[0,2]:.2f},{factory_K[1,2]:.2f}")
        print(f"           charuco fx,fy,cx,cy = "
              f"{K[0,0]:.2f},{K[1,1]:.2f},{K[0,2]:.2f},{K[1,2]:.2f}")
        print(f"           charuco D = {np.asarray(D).flatten()}")

        backup_path = overwrite_color_intrinsics(
            npz_path, backup_dir, K, D, result, serial)
        print(f"[SAVE] {npz_path} (color_K/color_D 교체)  factory backup -> {backup_path}")

        report["cameras"][str(cam_idx)] = {
            "serial": serial, "is_gripper": is_gripper, "status": "written",
            "rms_px": result["rms"], "num_views_used": result["n_used"],
            "num_views_total": result["n_total"], "num_dropped": result["n_dropped"],
            "K": np.asarray(K).tolist(), "D": np.asarray(D).flatten().tolist(),
            "factory_K": factory_K.tolist(),
            "factory_D": np.asarray(d0["color_D"]).flatten().tolist(),
        }

    cv2.destroyAllWindows()

    if not aborted:
        report_path = os.path.join(intr_dir, "charuco_intrinsics_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[SAVE] {report_path}")

        print("\n=== 요약 ===")
        for k, v in report["cameras"].items():
            st = v.get("status")
            if st == "written":
                print(f"  cam{k}: RMS {v['rms_px']:.3f}px  views {v['num_views_used']}  -> written")
            else:
                print(f"  cam{k}: {st}")
        print("[DONE] Step1b_charuco_intrinsics.py complete. "
              "이제 Step2~5는 갱신된 color_K/color_D 를 그대로 사용합니다.")
    else:
        print("[DONE] 중단됨.")


if __name__ == "__main__":
    main()
