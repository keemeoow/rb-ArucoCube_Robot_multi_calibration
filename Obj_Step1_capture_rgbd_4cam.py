# Obj_Step1_capture_rgbd_4cam.py
# 4대 RealSense RGBD 동시 캡처 — 물체 포즈추정용 (Step2_capture.py 의 큐브
# 캡처 → 객체 캡처 버전). 큐브/charuco 검출/게이트는 모두 제거되어 있고,
# 핵심만 남겼다:
#   - 4-캠 동기 캡처 + 라이브 quad 프리뷰
#   - 로봇 manual-robot 소켓 연동 (capture_pose_6dof 수신 → 자동 저장)
#   - meta.json 에 event_id / frame_id / capture_pose_6dof / 4x4 / joints 기록
#   - 그리퍼 캠 폴더에 T_base_ee_<frame>.npy 추가 저장 (pose pipeline 이 사용)
#
# 사용 예:
#   # (A) 로봇 서버 연동 (server/obj_robot_calb.py 와 짝)
#   #     ※ 로봇 서버를 먼저 실행 ("Server on port 12348. Waiting..." 확인) 후 아래 실행.
#   python Obj_Step1_capture_rgbd_4cam.py \
#       --save_dir ./data/object_capture --intrinsics_dir ./intrinsics \
#       --use_robot --robot_ip 192.168.0.23 --robot_port 12348 --show
#
#   # (B) 수동 SPACE 캡처 (TCP 없이; 그리퍼 캠은 pose pipeline 에서 사용 불가)
#   python Obj_Step1_capture_rgbd_4cam.py \
#       --save_dir ./data/object_capture --intrinsics_dir ./intrinsics
#
# 출력:
#   save_dir/
#     cam0/rgb_000000.jpg, depth_000000.png
#     cam1/...
#     cam2/...   (그리퍼 캠) + T_base_ee_000000.npy
#     cam3/...
#     quad/frame_000000.jpg
#     meta.json        : 캡처별 메타데이터 (event_id, TCP, joints, gates ...)
#     waypoints.json   : 수동 캡처(`c`) 시 누적되는 capture_joints/capture_tcp/
#                        station 등. 로봇 서버 `auto` 명령이 이 파일을 socket 으로
#                        받아 자동 재현 (replay).
#
# 로봇 연동 protocol (newline-delimited JSON):
#   - 로봇→PC: {command:capture, ..., is_replay:bool}      → PC 가 do_capture +
#              (is_replay=false 일 때만) waypoints.json append
#   - 로봇→PC: {command:request_waypoints}                  → PC 가
#              {status:ok, waypoints_data:{waypoints:[...]}} 응답
#   - 로봇→PC: {command:quit}                               → loop 종료

import os
import sys as _sys_top
import json
import time
import argparse
import select as _select
import socket as _sock
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera import RealSenseCamera
from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return {}, None
    with open(map_path, "r") as f:
        m = json.load(f)
    return m.get("serial_to_idx", {}), m.get("gripper_cam_idx", None)


# ─────────────────────────────────────────────────────────────────
# Preview helpers
# ─────────────────────────────────────────────────────────────────

def annotate_tile(img: np.ndarray, ci: int, is_gripper: bool,
                   frame_idx: int, ts_ms: Optional[float]) -> np.ndarray:
    out = img.copy()
    tag = "GRIP" if is_gripper else "FIX"
    color = (0, 200, 255) if is_gripper else (0, 255, 0)
    cv2.putText(out, f"cam{ci} [{tag}]", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(out, f"frame {frame_idx}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if ts_ms is not None:
        cv2.putText(out, f"ts={ts_ms:.0f}ms", (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return out


def make_quad_image(frames_dict: Dict[int, dict], cam_order: List[int],
                    gripper_cam_idx: Optional[int],
                    frame_idx: int) -> np.ndarray:
    """4개 카메라를 2x2 로 합쳐 라이브 프리뷰 이미지를 만든다."""
    tiles = []
    tile_h, tile_w = None, None
    for ci in cam_order:
        fr = frames_dict.get(ci)
        if fr is not None and fr.get("color") is not None:
            img = fr["color"]
            if tile_h is None:
                tile_h, tile_w = img.shape[:2]
            tiles.append(annotate_tile(
                img, ci,
                is_gripper=(gripper_cam_idx is not None and ci == gripper_cam_idx),
                frame_idx=frame_idx, ts_ms=fr.get("ts_ms")))
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
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    return cv2.vconcat([top, bottom])


def append_status_footer(img: np.ndarray, lines: List[str],
                         colors: List[Tuple[int, int, int]]) -> np.ndarray:
    pad = 26 * len(lines) + 14
    h, w = img.shape[:2]
    foot = np.zeros((pad, w, 3), dtype=np.uint8)
    y = 22
    for i, line in enumerate(lines):
        col = colors[i] if i < len(colors) else (220, 220, 220)
        cv2.putText(foot, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        y += 24
    return cv2.vconcat([img, foot])


# ─────────────────────────────────────────────────────────────────
# Capture core (manual + manual-robot 공용)
# ─────────────────────────────────────────────────────────────────

def grab_all(cams: Dict[int, RealSenseCamera]) -> Dict[int, dict]:
    out = {}
    for ci, cam in cams.items():
        color, depth, ts_ms = cam.get_latest()
        out[ci] = {"color": color, "depth": depth, "ts_ms": ts_ms}
    return out


def wait_for_start_command(cams, cam_order, gripper_cam_idx,
                            extra_lines: Optional[List[str]] = None) -> bool:
    """프리뷰 창을 띄우고 사용자가 터미널에서 'start' 입력할 때까지 대기.

    Returns: True 시작 / False 사용자가 q/ESC 또는 'quit' 입력해 종료 요청.
    """
    print("")
    print("=" * 60)
    print(" Live preview — type 'start' (then ENTER) in this terminal to begin")
    print(" or type 'quit' / press q in the preview window to abort")
    print("=" * 60)
    if extra_lines:
        for ln in extra_lines:
            print(" " + ln)

    start_event = threading.Event()
    quit_event = threading.Event()

    def stdin_reader():
        while not (start_event.is_set() or quit_event.is_set()):
            try:
                r, _, _ = _select.select([_sys_top.stdin], [], [], 0.2)
                if not r:
                    continue
                line = _sys_top.stdin.readline()
                if not line:
                    quit_event.set()
                    return
                token = line.strip().lower()
                if token == "start":
                    start_event.set()
                    return
                elif token in ("quit", "q", "exit"):
                    quit_event.set()
                    return
                elif token:
                    print(f"  type 'start' or 'quit' (got: {token!r})")
            except Exception:
                quit_event.set()
                return

    t = threading.Thread(target=stdin_reader, daemon=True)
    t.start()

    win = "Preview - waiting for 'start'"
    while not start_event.is_set() and not quit_event.is_set():
        live = grab_all(cams)
        if any(fr["color"] is not None for fr in live.values()):
            quad = make_quad_image(live, cam_order, gripper_cam_idx, 0)
            footer = ["[WAITING] Type 'start' + ENTER in terminal to begin"]
            colors = [(0, 200, 255)]
            if extra_lines:
                footer += extra_lines
                colors += [(220, 220, 220)] * len(extra_lines)
            quad = append_status_footer(quad, footer, colors)
            h2 = int(quad.shape[0] * 0.6); w2 = int(quad.shape[1] * 0.6)
            cv2.imshow(win, cv2.resize(quad, (w2, h2)))
        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key == ord('q'):
            quit_event.set()
            break

    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    if start_event.is_set():
        print("[start] confirmed, proceeding...")
        return True
    print("[abort] user cancelled before start.")
    return False


# ─────────────────────────────────────────────────────────────────
# Tier-1 quality gate (캡처 시점 cheap 검증; SAM 풀 검증은 오프라인)
# ─────────────────────────────────────────────────────────────────

def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────────────
# Profile-aware HSV color-prior coverage (Obj_pipeline_core 의 ColorPriorConfig
# 와 호환되는 JSON 만 직접 파싱; mobile_sam 등 무거운 의존 X)
# ─────────────────────────────────────────────────────────────────

def load_profile_color_priors(paths: List[str]) -> List[dict]:
    """profile JSON 들을 로드해서 [{name, hue_ref, hue_radius, s_min, v_min, v_max}, ...]
    형태로 반환. enabled=false 인 profile 은 skip."""
    out = []
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] profile 로드 실패: {p} ({e})")
            continue
        cp = data.get("color_prior") or {}
        if not cp.get("enabled", False):
            continue
        out.append({
            "name": data.get("name") or Path(p).stem,
            "hue_ref": float(cp.get("hue_ref", 0.0)),
            "hue_radius": float(cp.get("hue_radius", 12.0)),
            "s_min": int(cp.get("s_min", 100)),
            "v_min": int(cp.get("v_min", 70)),
            "v_max": int(cp.get("v_max", 255)),
        })
    return out


def _hue_dist(h: np.ndarray, ref: float) -> np.ndarray:
    """OpenCV H 채널은 0..179. 환형 거리."""
    d = np.abs(h.astype(np.int16) - int(ref))
    return np.minimum(d, 180 - d)


def hsv_coverage(color_bgr: np.ndarray, prior: dict) -> float:
    """profile color_prior 매치 픽셀 비율 (0..1)."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]; s = hsv[:, :, 1]; v = hsv[:, :, 2]
    hue_ok = _hue_dist(h, prior["hue_ref"]) <= prior["hue_radius"]
    sv_ok = (s >= prior["s_min"]) & (v >= prior["v_min"]) & (v <= prior["v_max"])
    mask = hue_ok & sv_ok
    return float(mask.sum()) / float(mask.size)


class GripperSamGate:
    """그리퍼 캠 한 장만 SAM center-prompt 로 추론해 빠르게 'object visible?' 검증.

    풀 SAM 파이프라인 (profile 별 prompt + bbox combine 등) 은 비용이 크므로,
    여기서는 단순 중심점 prompt → multimask 중 best score → 마스크 면적/스코어
    임계값 체크만 한다 (~50–500ms).

    Pass 조건:
      score > min_score  AND  min_area_pct ≤ mask_area / image_area ≤ max_area_pct
    """

    def __init__(self, weights_path: Optional[str] = None,
                 device: str = "cpu",
                 min_score: float = 0.85,
                 min_area_pct: float = 0.005,
                 max_area_pct: float = 0.50):
        # lazy import — gate 안 쓰면 mobile_sam 의존 X
        from mobile_sam import sam_model_registry, SamPredictor  # noqa
        if weights_path is None:
            here = Path(__file__).resolve().parent
            weights_path = str(here / "weights" / "mobile_sam.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"SAM weights not found: {weights_path}")
        sam = sam_model_registry["vit_t"](checkpoint=weights_path)
        sam.to(device); sam.eval()
        self._predictor = SamPredictor(sam)
        self.min_score = float(min_score)
        self.min_area_pct = float(min_area_pct)
        self.max_area_pct = float(max_area_pct)
        self.weights_path = weights_path
        self.device = device

    def evaluate(self, color_bgr: np.ndarray) -> Tuple[bool, dict]:
        """단일 이미지 검증. Returns (pass, metrics)."""
        if color_bgr is None:
            return False, {"reason": "no_image"}
        h, w = color_bgr.shape[:2]
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        try:
            self._predictor.set_image(rgb)
            cx, cy = w // 2, h // 2
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        except Exception as e:
            return False, {"reason": f"sam_error:{e}"}
        # best score 마스크 선택
        if scores is None or len(scores) == 0:
            return False, {"reason": "no_mask"}
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        score = float(scores[best_idx])
        area_pct = float(mask.sum()) / float(h * w)
        report = {
            "score": score,
            "area_pct": area_pct,
            "min_score": self.min_score,
            "min_area_pct": self.min_area_pct,
            "max_area_pct": self.max_area_pct,
        }
        reasons = []
        if score < self.min_score:
            reasons.append(f"score({score:.3f}<{self.min_score:.3f})")
        if area_pct < self.min_area_pct:
            reasons.append(f"area_small({area_pct:.4f}<{self.min_area_pct:.4f})")
        if area_pct > self.max_area_pct:
            reasons.append(f"area_huge({area_pct:.3f}>{self.max_area_pct:.3f})")
        report["reasons"] = reasons
        report["pass"] = (len(reasons) == 0)
        return report["pass"], report


def evaluate_quality_gate(frames: Dict[int, dict],
                           gripper_cam_idx: Optional[int],
                           cfg: dict,
                           color_priors: Optional[List[dict]] = None) -> Tuple[bool, dict]:
    """캠별 (blur, exposure, depth_coverage) + (옵션) profile-aware HSV coverage.
    모든 캠 pass + 적어도 N 개 profile 의 색 prior coverage ≥ 임계값일 때 True.

    cfg 키 (cheap signals):
      blur_min_fixed   : 고정 캠 Laplacian variance 최소값 (기본 80)
      blur_min_gripper : 그리퍼 캠 (close-up + 모션 블러 가능) 최소값 (기본 40)
      exposure_min     : 평균 밝기 0..255 최소 (기본 25)
      exposure_max     : 평균 밝기 최대 (기본 230)
      depth_min_mm     : 유효 depth 하한 (기본 200 = 20cm)
      depth_max_mm     : 유효 depth 상한 (기본 1500 = 1.5m)
      depth_cov_min    : 유효 depth 픽셀 비율 최소 (기본 0.20)
      check_depth      : depth 게이트 사용 여부 (기본 True)

    color_priors 가 주어지면 (profile_aware HSV):
      cfg["color_min_cov"]      : profile 별 coverage 임계값 (기본 0.005)
      cfg["color_min_objects"]  : 임계값 통과해야 하는 profile 최소 수 (기본 1)
      cfg["color_check_cams"]   : 어느 캠에서 검사할지 ("any"/"all"/"gripper")
                                   (기본 "any" — 한 캠이라도 만족하면 통과)
    """
    per_cam = {}
    fail_reasons = []
    for ci, fr in frames.items():
        c = fr.get("color")
        d = fr.get("depth")
        is_gripper = (gripper_cam_idx is not None and ci == gripper_cam_idx)
        cam_rec = {"is_gripper": is_gripper}
        if c is None:
            cam_rec["pass"] = False
            cam_rec["reason"] = "no_color"
            per_cam[int(ci)] = cam_rec
            fail_reasons.append("cam{}:no_color".format(ci))
            continue
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        blur = _laplacian_var(gray)
        mean_v = float(c.mean())
        cam_rec["blur"] = blur
        cam_rec["mean"] = mean_v
        blur_min = cfg["blur_min_gripper"] if is_gripper else cfg["blur_min_fixed"]
        if blur < blur_min:
            cam_rec["pass"] = False
            cam_rec["reason"] = "blurry({:.1f}<{:.0f})".format(blur, blur_min)
            per_cam[int(ci)] = cam_rec
            fail_reasons.append("cam{}:blurry".format(ci))
            continue
        if mean_v < cfg["exposure_min"] or mean_v > cfg["exposure_max"]:
            cam_rec["pass"] = False
            cam_rec["reason"] = "exposure({:.1f})".format(mean_v)
            per_cam[int(ci)] = cam_rec
            fail_reasons.append("cam{}:exposure".format(ci))
            continue
        if cfg.get("check_depth", True):
            if d is None:
                cam_rec["pass"] = False
                cam_rec["reason"] = "no_depth"
                per_cam[int(ci)] = cam_rec
                fail_reasons.append("cam{}:no_depth".format(ci))
                continue
            valid = ((d > cfg["depth_min_mm"]) & (d < cfg["depth_max_mm"]))
            cov = float(valid.sum()) / float(d.size)
            cam_rec["depth_cov"] = cov
            if cov < cfg["depth_cov_min"]:
                cam_rec["pass"] = False
                cam_rec["reason"] = "depth_sparse({:.2f}<{:.2f})".format(cov, cfg["depth_cov_min"])
                per_cam[int(ci)] = cam_rec
                fail_reasons.append("cam{}:depth_sparse".format(ci))
                continue
        cam_rec["pass"] = True
        cam_rec["reason"] = "ok"
        per_cam[int(ci)] = cam_rec

    all_pass = all(rec.get("pass") for rec in per_cam.values())

    # profile-aware HSV coverage
    color_report: Optional[dict] = None
    if color_priors and all_pass:
        check_cams = cfg.get("color_check_cams", "any")
        min_cov = float(cfg.get("color_min_cov", 0.005))
        min_objects = int(cfg.get("color_min_objects", 1))
        per_cam_color: Dict[int, dict] = {}
        objects_passed_globally = set()
        for ci, fr in frames.items():
            c = fr.get("color")
            if c is None:
                continue
            is_gripper = (gripper_cam_idx is not None and ci == gripper_cam_idx)
            if check_cams == "gripper" and not is_gripper:
                continue
            cam_obj_cov = {}
            for prior in color_priors:
                cov = hsv_coverage(c, prior)
                cam_obj_cov[prior["name"]] = cov
                if cov >= min_cov and check_cams != "all":
                    objects_passed_globally.add(prior["name"])
            per_cam_color[int(ci)] = cam_obj_cov

        # check_cams == "all": 모든 캠에서 같은 profile 이 임계값 통과해야 함
        if check_cams == "all":
            obj_names = [p["name"] for p in color_priors]
            for nm in obj_names:
                if all(nm in pc and pc[nm] >= min_cov
                        for pc in per_cam_color.values()):
                    objects_passed_globally.add(nm)

        n_passed = len(objects_passed_globally)
        color_pass = (n_passed >= min_objects)
        color_report = {
            "check_cams": check_cams,
            "min_cov": min_cov,
            "min_objects": min_objects,
            "n_objects_pass": n_passed,
            "objects_passed": sorted(list(objects_passed_globally)),
            "per_cam": per_cam_color,
            "pass": color_pass,
        }
        if not color_pass:
            all_pass = False
            fail_reasons.append(
                "color_prior({}/{} objects)".format(n_passed, min_objects))

    out: dict = {"per_cam": per_cam, "reasons": fail_reasons}
    if color_report is not None:
        out["color"] = color_report
    return all_pass, out


def write_frame(root: str, frame_idx: int, frames: Dict[int, dict],
                gripper_cam_idx: Optional[int],
                T_base_ee: Optional[np.ndarray],
                save_depth: bool) -> Optional[dict]:
    """모든 캠이 valid 일 때만 저장. 성공 시 cap_rec dict (메타용) 반환."""
    if not all(fr["color"] is not None for fr in frames.values()):
        return None
    if save_depth and not all(fr["depth"] is not None for fr in frames.values()):
        return None

    fid_str = f"{frame_idx:06d}"
    cam_records: Dict[str, dict] = {}
    for ci in sorted(frames.keys()):
        fr = frames[ci]
        rgb_rel = f"cam{ci}/rgb_{fid_str}.jpg"
        cv2.imwrite(os.path.join(root, rgb_rel), fr["color"])
        depth_rel = None
        if save_depth and fr["depth"] is not None:
            depth_rel = f"cam{ci}/depth_{fid_str}.png"
            cv2.imwrite(os.path.join(root, depth_rel), fr["depth"])
        cam_rec = {
            "saved": True,
            "is_gripper": (gripper_cam_idx is not None and ci == gripper_cam_idx),
            "rgb_path": rgb_rel,
            "depth_path": depth_rel,
            "ts_ms": fr.get("ts_ms"),
        }
        if (gripper_cam_idx is not None and ci == gripper_cam_idx
                and T_base_ee is not None):
            ee_rel = f"cam{ci}/T_base_ee_{fid_str}.npy"
            np.save(os.path.join(root, ee_rel), T_base_ee.astype(np.float64))
            cam_rec["T_base_ee_path"] = ee_rel
        cam_records[str(ci)] = cam_rec
    return {"frame_id": fid_str, "cams": cam_records}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Object capture: 4-cam RGBD + per-frame TCP")
    parser.add_argument("--save_dir", required=True, help="저장 폴더")
    parser.add_argument("--intrinsics_dir", default="./intrinsics",
                        help="device_map.json 위치")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--save_depth", action="store_true", default=True,
                        help="depth 저장 (기본 on)")
    parser.add_argument("--no_depth", dest="save_depth", action="store_false")
    parser.add_argument("--show", action="store_true",
                        help="라이브 quad 프리뷰 창 표시")
    # 로봇 서버 연동
    parser.add_argument("--use_robot", action="store_true",
                        help="로봇 서버 연동 (manual-robot 모드)")
    parser.add_argument("--robot_ip", default=None)
    parser.add_argument("--robot_port", type=int, default=None)
    # 그리퍼 캠 강제 지정 (device_map.json 에 없을 때)
    parser.add_argument("--gripper_cam_idx", type=int, default=None)
    # Tier-1 quality gate (cheap signals; SAM 풀 검증은 오프라인)
    parser.add_argument("--quality_gate", action="store_true",
                        help="캡처 시점 quality gate 활성화 (blur/exposure/depth)")
    parser.add_argument("--gate_blur_fixed", type=float, default=80.0)
    parser.add_argument("--gate_blur_gripper", type=float, default=40.0)
    parser.add_argument("--gate_expo_min", type=float, default=25.0)
    parser.add_argument("--gate_expo_max", type=float, default=230.0)
    parser.add_argument("--gate_depth_min_mm", type=float, default=200.0)
    parser.add_argument("--gate_depth_max_mm", type=float, default=1500.0)
    parser.add_argument("--gate_depth_cov", type=float, default=0.20)
    # Tier-1 profile-aware HSV color-prior coverage (옵션)
    parser.add_argument("--profiles", default=None,
                        help="profile JSON 경로 (콤마구분) — color_prior coverage 검사용")
    parser.add_argument("--profiles_dir", default=None,
                        help="profile JSON 들이 들어있는 폴더 — *.json 모두 사용")
    parser.add_argument("--gate_color_min_cov", type=float, default=0.005,
                        help="profile color_prior 매치 픽셀 비율 임계값 (기본 0.005=0.5%)")
    parser.add_argument("--gate_color_min_objects", type=int, default=1,
                        help="임계값 통과해야 하는 profile 최소 수 (기본 1)")
    parser.add_argument("--gate_color_check_cams", default="any",
                        choices=["any", "all", "gripper"],
                        help="any: 어느 캠이든 OK / all: 모든 캠 OK / gripper: 그리퍼만")
    # 그리퍼 캠 한정 SAM gate (옵션, ~50-500ms/캡처)
    parser.add_argument("--sam_gate", choices=["none", "gripper"], default="none",
                        help="그리퍼 캠 한 장만 SAM 으로 'object visible?' 검증")
    parser.add_argument("--sam_weights", default=None,
                        help="MobileSAM .pt 경로 (기본: <script>/weights/mobile_sam.pt)")
    parser.add_argument("--sam_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sam_min_score", type=float, default=0.85)
    parser.add_argument("--sam_min_area_pct", type=float, default=0.005)
    parser.add_argument("--sam_max_area_pct", type=float, default=0.50)
    # 시작 게이트: cv2 프리뷰 띄우고 터미널에서 'start' 입력 시까지 대기
    parser.add_argument("--no_start_gate", action="store_true",
                        help="기본은 cv2 프리뷰 + 'start' 입력 대기. 이 플래그 시 즉시 시작.")
    args = parser.parse_args()

    quality_cfg = {
        "blur_min_fixed":   args.gate_blur_fixed,
        "blur_min_gripper": args.gate_blur_gripper,
        "exposure_min":     args.gate_expo_min,
        "exposure_max":     args.gate_expo_max,
        "depth_min_mm":     args.gate_depth_min_mm,
        "depth_max_mm":     args.gate_depth_max_mm,
        "depth_cov_min":    args.gate_depth_cov,
        "check_depth":      bool(args.save_depth),
        "color_min_cov":     args.gate_color_min_cov,
        "color_min_objects": args.gate_color_min_objects,
        "color_check_cams":  args.gate_color_check_cams,
    }

    # profile-aware HSV color priors (옵션)
    color_priors: List[dict] = []
    profile_paths: List[str] = []
    if args.profiles:
        profile_paths += [p.strip() for p in args.profiles.split(",") if p.strip()]
    if args.profiles_dir:
        for p in sorted(Path(args.profiles_dir).glob("*.json")):
            profile_paths.append(str(p))
    if profile_paths:
        color_priors = load_profile_color_priors(profile_paths)
        if color_priors:
            print(f"[INFO] color-prior gate: {len(color_priors)} profile(s) "
                  f"({', '.join(p['name'] for p in color_priors)})")
        else:
            print("[INFO] profile 들 중 color_prior.enabled=true 인 항목 없음")

    if args.use_robot and (not args.robot_ip or not args.robot_port):
        parser.error("--use_robot 사용 시 --robot_ip, --robot_port 필요")

    # SAM gate 초기화 (옵션) — 가능한 빨리 실패하도록 args 검증 직후
    sam_gate: Optional[GripperSamGate] = None
    if args.sam_gate == "gripper":
        try:
            sam_gate = GripperSamGate(
                weights_path=args.sam_weights,
                device=args.sam_device,
                min_score=args.sam_min_score,
                min_area_pct=args.sam_min_area_pct,
                max_area_pct=args.sam_max_area_pct,
            )
            print(f"[INFO] SAM gripper-gate enabled "
                  f"(device={args.sam_device}, weights={sam_gate.weights_path})")
        except Exception as e:
            print(f"[ERROR] SAM gripper-gate 초기화 실패: {e}")
            return

    save_dir = ensure_dir(args.save_dir)
    quad_dir = ensure_dir(os.path.join(save_dir, "quad"))
    meta_path = os.path.join(save_dir, "meta.json")

    # ── 카메라 탐색 + 인덱스 매핑 ──
    devs = RealSenseCamera.list_devices()
    if not devs:
        raise RuntimeError("RealSense 카메라가 연결되어 있지 않습니다.")
    print(f"[INFO] 감지된 카메라 {len(devs)}대:")
    for s, n in devs.items():
        print(f"  {s}  ({n})")

    serial_to_idx, dm_gripper = load_device_map(args.intrinsics_dir)
    gripper_cam_idx = (args.gripper_cam_idx
                       if args.gripper_cam_idx is not None else dm_gripper)
    if serial_to_idx:
        idx_serial = []
        for serial in devs.keys():
            if serial in serial_to_idx:
                idx_serial.append((int(serial_to_idx[serial]), serial))
            else:
                print(f"[WARN] device_map.json에 없는 시리얼: {serial}")
        idx_serial.sort(key=lambda x: x[0])
    else:
        print("[WARN] device_map.json 없음 -> 시리얼 정렬 순서 사용")
        idx_serial = [(i, s) for i, s in enumerate(sorted(devs.keys()))]
    if not idx_serial:
        raise RuntimeError("사용 가능한 카메라가 없습니다.")

    if gripper_cam_idx is None:
        print("[WARN] 그리퍼 캠 id 미정의 — T_base_ee 미저장. "
              "Pose pipeline 은 모든 캠을 정적으로 취급함.")
    else:
        print(f"[INFO] gripper cam: cam{gripper_cam_idx}")
    if args.use_robot and gripper_cam_idx is None:
        print("[ERROR] --use_robot 모드는 gripper_cam_idx 가 필요. "
              "device_map.json 에 'gripper_cam_idx' 추가 또는 "
              "--gripper_cam_idx 로 지정하세요.")
        return

    # ── 카메라 시작 ──
    RealSenseCamera.reset_all_devices()
    cams: Dict[int, RealSenseCamera] = {}
    for ci, serial in idx_serial:
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
        ensure_dir(os.path.join(save_dir, f"cam{ci}"))

    cam_order = sorted(cams.keys())
    print(f"\n[INFO] {len(cams)}대 카메라 시작 완료 "
          f"({args.width}x{args.height} @ {args.fps}fps, RGBD)")

    # ── meta.json 초기화/이어쓰기 ──
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "captures" not in meta:
            meta["captures"] = []
    else:
        meta = {
            "session_mode": "object_capture",
            "gripper_cam_idx": gripper_cam_idx,
            "cam_order": cam_order,
            "image_size": [args.width, args.height],
            "fps": args.fps,
            "save_depth": bool(args.save_depth),
            "captures": [],
        }
    # 기존 frame index 이어쓰기
    event_id = 0
    if meta["captures"]:
        try:
            event_id = max(int(c.get("event_id", -1)) for c in meta["captures"]) + 1
        except Exception:
            event_id = len(meta["captures"])
    print(f"[INFO] starting event_id = {event_id}")

    # ── start 게이트: cv2 프리뷰 + 사용자 'start' 입력 대기 ──
    if not args.no_start_gate:
        extra = []
        if args.use_robot:
            extra.append(f"will connect to robot {args.robot_ip}:{args.robot_port}")
        if args.quality_gate:
            extra.append("Tier-1 quality gate ON")
        if color_priors:
            extra.append(f"color-prior gate ON ({len(color_priors)} profiles)")
        if sam_gate is not None:
            extra.append("gripper SAM gate ON")
        ok_start = wait_for_start_command(cams, cam_order, gripper_cam_idx, extra)
        if not ok_start:
            for cam in cams.values():
                cam.stop()
            cv2.destroyAllWindows()
            return

    def persist_meta():
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── do_capture ──
    def do_capture(capture_pose_6dof: Optional[List[float]] = None,
                   pose_index: Optional[int] = None,
                   robot_joints_6dof: Optional[List[float]] = None,
                   ) -> Tuple[bool, str]:
        nonlocal event_id
        frames = grab_all(cams)
        # 캠 valid 검증
        bad = [ci for ci, fr in frames.items() if fr["color"] is None]
        if bad:
            return False, f"invalid_color_cams={bad}"
        if args.save_depth:
            bad_d = [ci for ci, fr in frames.items() if fr["depth"] is None]
            if bad_d:
                return False, f"invalid_depth_cams={bad_d}"

        # Tier-1 quality gate (옵션) — 통과 못하면 저장 skip
        gate_report = None
        if args.quality_gate:
            ok_gate, gate_report = evaluate_quality_gate(
                frames, gripper_cam_idx, quality_cfg,
                color_priors=color_priors or None)
            if not ok_gate:
                return False, "quality_gate:" + ",".join(gate_report["reasons"])

        # 그리퍼 캠 SAM gate (옵션)
        sam_report = None
        if sam_gate is not None and gripper_cam_idx is not None:
            g = frames.get(gripper_cam_idx, {})
            ok_sam, sam_report = sam_gate.evaluate(g.get("color"))
            if not ok_sam:
                return False, "sam_gate:" + ",".join(sam_report.get("reasons", []))

        T_base_ee = None
        if capture_pose_6dof is not None:
            try:
                T_base_ee = euler_deg_to_matrix(*[float(x) for x in capture_pose_6dof])
            except Exception as e:
                print(f"[WARN] capture_pose_6dof → 4x4 변환 실패: {e}")

        rec = write_frame(save_dir, event_id, frames,
                          gripper_cam_idx, T_base_ee, args.save_depth)
        if rec is None:
            return False, "write_frame_failed"

        # 메타 채우기
        cap_rec = {"event_id": event_id, **rec, "pose_index": pose_index}
        if gate_report is not None:
            cap_rec["quality_gate"] = gate_report
        if sam_report is not None:
            cap_rec["sam_gate"] = sam_report
        if capture_pose_6dof is not None:
            tcp_f = [float(x) for x in capture_pose_6dof]
            cap_rec["capture_pose_6dof"] = tcp_f
            cap_rec["robot_pose_6dof"] = tcp_f          # Step2/3 호환 키
            if T_base_ee is not None:
                T_list = T_base_ee.tolist()
                cap_rec["capture_pose_matrix_4x4"] = T_list
                cap_rec["robot_pose_matrix_4x4"] = T_list
        if robot_joints_6dof is not None:
            cap_rec["robot_joints_6dof"] = [float(x) for x in robot_joints_6dof]

        meta["captures"].append(cap_rec)
        persist_meta()

        # quad 저장
        quad = make_quad_image(frames, cam_order, gripper_cam_idx, event_id)
        cv2.imwrite(os.path.join(quad_dir, f"frame_{event_id:06d}.jpg"), quad)

        msg = f"[SAVE] event={event_id} ({len(frames)}cams)"
        if T_base_ee is not None:
            t = T_base_ee[:3, 3]
            msg += f" TCP=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]m"
        elif gripper_cam_idx is not None:
            msg += " (no TCP — gripper cam unusable)"
        print(msg)
        event_id += 1
        return True, "ok"

    # ── Mode A: 로봇 서버 연동 (manual-robot) ──
    if args.use_robot:
        print(f"\n[MODE] manual-robot — connecting {args.robot_ip}:{args.robot_port}")
        sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        sock.settimeout(None)
        sock.connect((args.robot_ip, int(args.robot_port)))
        print(f"[ManualRobot] Connected")

        # ── Waypoints accumulator (수동 capture 시 누적, auto replay 의 source) ──
        waypoints_path = os.path.join(save_dir, "waypoints.json")
        accumulated_waypoints: List[dict] = []
        if os.path.exists(waypoints_path):
            try:
                with open(waypoints_path, "r") as f:
                    _wpd = json.load(f)
                if isinstance(_wpd, dict) and isinstance(_wpd.get("waypoints"), list):
                    accumulated_waypoints = _wpd["waypoints"]
                    print(f"[INFO] loaded {len(accumulated_waypoints)} existing waypoints "
                          f"from {waypoints_path}")
            except Exception as e:
                print(f"[WARN] failed to load existing waypoints.json: {e}")

        def persist_waypoints():
            try:
                with open(waypoints_path, "w") as f:
                    json.dump({"waypoints": accumulated_waypoints}, f, indent=2)
            except Exception as e:
                print(f"[WARN] waypoints.json 저장 실패: {e}")

        last_status_lines: List[str] = [
            "waiting for capture...",
            "preview keys: q quit | x soft stop (rb.stop) | X hard abort (rb.abort)",
        ]

        # ── Newline-delimited JSON framing (로봇 서버와 통일) ──
        # cv2 (Qt backend) GUI 는 메인 스레드에서만 허용되므로 preview/소켓 polling
        # 을 한 루프에서 같이 돌린다. recv 는 select 로 non-blocking 폴링.
        recv_buf = bytearray()

        def send_one(obj):
            sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))

        def try_read_message():
            """Non-blocking poll. Returns (msg_dict_or_None, disconnected_bool)."""
            # 1) 버퍼에 이미 완성된 메시지가 있으면 바로 반환.
            if b'\n' in recv_buf:
                idx = recv_buf.index(b'\n')
                line = bytes(recv_buf[:idx])
                del recv_buf[:idx + 1]
                return json.loads(line.decode("utf-8").strip()), False
            # 2) 소켓에 읽을 게 있을 때만 recv (select 0-timeout).
            ready, _, _ = _select.select([sock], [], [], 0)
            if not ready:
                return None, False
            try:
                chunk = sock.recv(65536)
            except Exception:
                return None, False
            if not chunk:
                return None, True  # peer closed
            recv_buf.extend(chunk)
            if b'\n' in recv_buf:
                idx = recv_buf.index(b'\n')
                line = bytes(recv_buf[:idx])
                del recv_buf[:idx + 1]
                return json.loads(line.decode("utf-8").strip()), False
            return None, False  # 아직 메시지 미완성

        def send_action(action_name: str):
            try:
                send_one({"action": action_name})
                print(f"[client] sent action={action_name!r} to server")
            except Exception as e:
                print(f"[client] {action_name} send 실패: {e}")

        if args.show:
            print("[INFO] Live preview started (main-thread; q quit / x soft / X hard)")

        win = "Object Capture (4 cams)"
        try:
            while True:
                # ── (1) Live preview frame (main thread, Qt 호환) ──
                if args.show:
                    live = grab_all(cams)
                    if any(fr["color"] is not None for fr in live.values()):
                        quad = make_quad_image(live, cam_order, gripper_cam_idx, event_id)
                        quad = append_status_footer(
                            quad, last_status_lines,
                            [(0, 255, 0)] + [(220, 220, 220)] * (len(last_status_lines) - 1))
                        h2 = int(quad.shape[0] * 0.6); w2 = int(quad.shape[1] * 0.6)
                        cv2.imshow(win, cv2.resize(quad, (w2, h2)))
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27 or key == ord('q'):
                        print("[client] preview closed by user — disconnecting")
                        break
                    if key == ord('x'):
                        send_action("stop")
                    elif key == ord('X'):
                        send_action("abort")
                else:
                    time.sleep(0.05)

                # ── (2) 소켓 메시지 polling (non-blocking) ──
                try:
                    msg, disconnected = try_read_message()
                except Exception as e:
                    print(f"[WARN] recv error: {e}")
                    break
                if disconnected:
                    print("[ManualRobot] Server disconnected.")
                    break
                if msg is None:
                    continue  # 다음 preview frame 으로

                cmd = msg.get("command", "")
                if cmd == "quit":
                    print("[ManualRobot] Server sent quit.")
                    break

                # ── 자동 모드 요청: 로봇이 waypoints.json 내용을 요구 ──
                if cmd == "request_waypoints":
                    send_one({
                        "status": "ok",
                        "waypoints_data": {"waypoints": accumulated_waypoints},
                    })
                    print(f"[ManualRobot] sent {len(accumulated_waypoints)} waypoints to robot "
                          f"(auto replay source)")
                    continue

                if cmd != "capture":
                    print(f"[ManualRobot] (ignored) command={cmd!r}")
                    continue

                capture_tcp = msg.get("capture_pose_6dof")
                pose_idx = msg.get("pose_index", event_id)
                r_joints = msg.get("robot_joints_6dof")
                is_replay = bool(msg.get("is_replay", False))
                viewpoint_name = msg.get("viewpoint_name")
                capture_mode_msg = msg.get("capture_mode")
                gripper_state_msg = msg.get("gripper_state")

                mode_label = "AUTO replay" if is_replay else "MANUAL"
                print(f"\n[ManualRobot:{mode_label}] capture (pose_index={pose_idx})")
                if capture_tcp:
                    print(f"  TCP: {capture_tcp}")

                ok, reason = do_capture(
                    capture_pose_6dof=capture_tcp,
                    pose_index=pose_idx,
                    robot_joints_6dof=r_joints,
                )

                # 수동 모드에서만 waypoints.json 에 append (replay 중복 방지).
                # 물체 고정 sweep 이므로 station/set/place 필드 없음.
                if ok and not is_replay and r_joints is not None:
                    wp = {
                        "pose_index": int(pose_idx) if pose_idx is not None else None,
                        "viewpoint_name": viewpoint_name,
                        "capture_joints": [float(x) for x in r_joints],
                        "capture_tcp": [float(x) for x in capture_tcp] if capture_tcp else None,
                        "capture_mode": capture_mode_msg,
                        "gripper_state": gripper_state_msg,
                    }
                    accumulated_waypoints.append(wp)
                    persist_waypoints()
                    print(f"  [waypoints] appended → total {len(accumulated_waypoints)} "
                          f"({waypoints_path})")

                last_status_lines = [
                    f"last event {event_id-1 if ok else event_id}: "
                    f"{'OK' if ok else 'SKIP'} ({reason})"
                ]
                send_one({
                    "action": "captured",
                    "status": "success" if ok else "skipped",
                    "reason": None if ok else reason,
                })
        finally:
            sock.close()
            for cam in cams.values():
                cam.stop()
            cv2.destroyAllWindows()
            persist_meta()
            persist_waypoints()
            print(f"\n[INFO] saved {len(meta['captures'])} captures → {os.path.abspath(save_dir)}")
            print(f"[INFO] saved {len(accumulated_waypoints)} waypoints → {waypoints_path}")
        return

    # ── Mode B: 수동 SPACE 캡처 ──
    print("\n조작 (수동 모드):")
    print("  SPACE : 현재 프레임 저장 (TCP 없음)")
    print("  s     : 연속 저장 모드 토글")
    print("  ESC/q : 종료\n")

    continuous = False
    try:
        while True:
            frames = grab_all(cams)
            quad = make_quad_image(frames, cam_order, gripper_cam_idx, event_id)
            footer = ["[REC ON]" if continuous else "[manual]",
                       "SPACE save | s toggle | q quit"]
            colors = [((0, 0, 255) if continuous else (0, 255, 0)), (220, 220, 220)]
            quad = append_status_footer(quad, footer, colors)
            h2 = int(quad.shape[0] * 0.6); w2 = int(quad.shape[1] * 0.6)
            cv2.imshow("Object Capture (4 cams)", cv2.resize(quad, (w2, h2)))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('s'):
                continuous = not continuous
                print(f"[INFO] 연속 저장 모드: {'ON' if continuous else 'OFF'}")

            do_save = (key == 32) or continuous
            if not do_save:
                continue
            ok, reason = do_capture()  # no TCP
            if not ok:
                print(f"[SKIP] event={event_id} reason={reason}")
    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()
        persist_meta()
        print(f"\n[INFO] saved {len(meta['captures'])} captures → {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
