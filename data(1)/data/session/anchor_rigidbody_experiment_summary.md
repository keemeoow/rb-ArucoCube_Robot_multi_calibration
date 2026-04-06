# Anchor / Rigid-Basis Experiment Summary

## 실험 1: consistency-aware anchor only
- calibration: `calib_out_anchor_consistent_fixed13`
- cube config: `cube_config_explicit_marker_pose_fixed13.json`

결과:
- `T_base_O` method: `consistent_event_anchor`
- anchor support: `4/18 events`
- anchor stability: `0.74 mm / 0.293 deg`
- cross-camera: `13.22 mm`
- reprojection: `1.204 px` (`PASS`)

해석:
- cube anchor 자체는 더 깔끔해졌다.
- 하지만 전체 cross-camera 수치는 기존 `calib_out_cam3_tracefix`와 사실상 동일하다.
- 즉 anchor refinement는 `T_base_O`의 해석 품질을 올렸지만, 현재 검증 지표의 주병목은 아니었다.

## 실험 2: consistency-aware anchor + rigid-basis refinement
- calibration: `calib_out_anchor_rigidbody_refined`
- cube config: `cube_config_explicit_marker_pose_refined_anchor.json`

결과:
- `T_base_O` method: `consistent_event_anchor`
- anchor support: `4/18 events`
- anchor stability: `0.11 mm / 0.013 deg`
- cross-camera: `12.15 mm`
- reprojection: `4.820 px` (`FAIL`)

해석:
- cross-camera는 `13.22 -> 12.15 mm`로 약간 좋아졌다.
- 하지만 reprojection이 `1.204 -> 4.820 px`로 크게 악화되었다.
- 따라서 현재 rigid-basis refinement는 event consistency를 일부 개선하지만, image-plane geometry를 지나치게 희생한다.

## 현재 권장본
현재 가장 균형이 좋은 결과는 `calib_out_cam3_tracefix`다.

이유:
- cross-camera: `13.22 mm`
- reprojection: `1.204 px` (`PASS`)
- hand-eye: `PASS`
- cam1/cam3 family selection 문제는 이미 verify/report 단계에서 상당 부분 정리됨

## 한 줄 결론
- `consistency-aware anchor`: 채택 가능
- `marker rigid-basis refinement`: 현재 버전은 보류
