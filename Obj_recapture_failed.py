#!/usr/bin/env python3
"""실패한 프레임만 골라 재촬영 waypoints JSON 을 생성한다.

사용 흐름:
  1) 첫 촬영 (수동 또는 auto) → meta.json + object_fixed_waypoints_station<NN>.json
  2) Obj_run_pipeline.py 실행 → pose/<frame_id>/quality_report.json 들 생성
  3) 이 스크립트로 fail frame 만 추려 waypoints 파일 생성
       python Obj_recapture_failed.py \
         --pose_dir ./output/pipeline_<ts>/pose \
         --meta ./data/object_capture/meta.json \
         --waypoints ./object_fixed_waypoints_station00.json \
         --out ./recapture_waypoints.json
  4) 서버에서 그 파일로 replay
       > auto recapture_waypoints.json 30
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set


def collect_failed_frame_ids(pose_dir: Path) -> List[str]:
    """pose_dir/<frame_id>/quality_report.json 을 모두 읽어 overall_pass=false 인
    frame_id 리스트를 반환."""
    failed: List[str] = []
    for sub in sorted(pose_dir.iterdir()):
        if not sub.is_dir():
            continue
        rpt_path = sub / "quality_report.json"
        if not rpt_path.exists():
            continue
        try:
            with open(rpt_path, "r") as f:
                rpt = json.load(f)
        except Exception as e:
            print(f"[WARN] 읽기 실패 {rpt_path}: {e}")
            continue
        if not rpt.get("overall_pass", False):
            fid = rpt.get("frame_id") or sub.name
            failed.append(str(fid))
    return failed


def map_frame_to_pose_index(meta_path: Path) -> Dict[str, Optional[int]]:
    """meta.json 의 captures 를 읽어 {frame_id: pose_index} 매핑.
    pose_index 가 없으면 event_id 를 사용."""
    with open(meta_path, "r") as f:
        meta = json.load(f)
    out: Dict[str, Optional[int]] = {}
    for cap in meta.get("captures", []):
        fid = cap.get("frame_id")
        if fid is None:
            ev = cap.get("event_id")
            if ev is None:
                continue
            fid = f"{int(ev):06d}"
        pose_idx = cap.get("pose_index")
        if pose_idx is None:
            pose_idx = cap.get("event_id")
        out[str(fid)] = (None if pose_idx is None else int(pose_idx))
    return out


def filter_waypoints(waypoints: List[dict],
                      failed_pose_indices: Set[int]) -> List[dict]:
    """waypoint 들 중 pose_index 가 failed set 에 속하는 항목만 추출."""
    kept = []
    for wp in waypoints:
        pi = wp.get("pose_index")
        if pi is None:
            continue
        if int(pi) in failed_pose_indices:
            kept.append(wp)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_dir", required=True,
                    help="pose/ 폴더 (frame_id 별 quality_report.json 가 들어있음)")
    ap.add_argument("--meta", required=True,
                    help="object_capture/meta.json 경로")
    ap.add_argument("--waypoints", required=True,
                    help="원본 object_fixed_waypoints_station<NN>.json 경로")
    ap.add_argument("--out", required=True,
                    help="필터링된 waypoints 저장 경로")
    ap.add_argument("--include_objects_without_pose", action="store_true",
                    help="pose 자체가 안 잡힌 (no pose) 프레임도 포함 (기본 포함)")
    ap.add_argument("--include_viewpoint_only_fail", action="store_true",
                    help="viewpoint diversity 만 fail 인 frame 도 재촬영 대상에 포함")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir)
    meta_path = Path(args.meta)
    wp_path = Path(args.waypoints)
    out_path = Path(args.out)

    # ── 1) failed frame_id 수집 ──
    failed_frame_ids = collect_failed_frame_ids(pose_dir)
    if not failed_frame_ids:
        print(f"[OK] {pose_dir} — 실패한 frame 없음. 출력 파일 미생성.")
        return
    print(f"[INFO] failed frames: {len(failed_frame_ids)}")
    for fid in failed_frame_ids[:10]:
        print(f"  - {fid}")
    if len(failed_frame_ids) > 10:
        print(f"  ... +{len(failed_frame_ids) - 10} more")

    # ── 2) frame_id → pose_index 매핑 ──
    fid_to_pose = map_frame_to_pose_index(meta_path)
    failed_pose_indices: Set[int] = set()
    unmapped: List[str] = []
    for fid in failed_frame_ids:
        pi = fid_to_pose.get(fid)
        if pi is None:
            unmapped.append(fid)
        else:
            failed_pose_indices.add(int(pi))
    if unmapped:
        print(f"[WARN] meta.json 에서 못 찾은 frame_id: {unmapped[:5]} "
              f"(+{max(0,len(unmapped)-5)} more)")
    if not failed_pose_indices:
        print("[ERROR] 매핑된 pose_index 가 없습니다. meta.json 의 captures 에 "
              "pose_index 또는 event_id 가 있는지 확인하세요.")
        return
    print(f"[INFO] failed pose_indices: {sorted(failed_pose_indices)[:20]}"
          f"{' ...' if len(failed_pose_indices) > 20 else ''}")

    # ── 3) 원본 waypoints 에서 매칭되는 항목만 추출 ──
    with open(wp_path, "r") as f:
        plan = json.load(f)
    wps = plan.get("waypoints") or []
    if not isinstance(wps, list):
        print("[ERROR] waypoints JSON 에 'waypoints' 리스트가 없습니다.")
        return
    kept = filter_waypoints(wps, failed_pose_indices)
    if not kept:
        print(f"[ERROR] {wp_path} 의 waypoints 중 failed pose_index 와 "
              f"매칭되는 항목이 없습니다.")
        return

    out_plan = dict(plan)
    out_plan["waypoints"] = kept
    out_plan["recapture_source_pose_dir"] = str(pose_dir)
    out_plan["recapture_failed_frame_ids"] = failed_frame_ids
    out_plan["recapture_kept_pose_indices"] = sorted(
        int(w["pose_index"]) for w in kept if w.get("pose_index") is not None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_plan, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 저장: {out_path}  ({len(kept)} / {len(wps)} waypoints)")
    print(f"     서버에서 replay: > auto {out_path.name} 30")


if __name__ == "__main__":
    main()
