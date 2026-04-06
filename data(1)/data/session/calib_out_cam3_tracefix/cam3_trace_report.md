# cam3 / T_base_C3 Trace Report

## 결론
`cam3`가 흔들린 직접 원인은 `cam3`의 검출 노이즈가 아니라, `Step3`의 base 통합 단계가 `cam3`를 잘못된 cube pose family에 붙이고 있었기 때문이다.

기존 로직은 `cube_anchor -> cam3`만 사용했다. 이 anchor는 raw event 추적 결과 `event 3~6`에서만 지지되었고, `cam0: id3 / ippe0`, `cam1: id1 / ippe0`라는 매우 좁은 marker family로 형성되었다.

반면 `cam3`는 전체 18개 공통 이벤트 중 14개에서 `cam0` 기준 상대자세(`T_C0_C3`)와 일관된 다른 family를 보였다. 따라서 기존 `T_base_C3`는 anchor 내부에서는 안정적이어도, 전체 base 통합에서는 다른 카메라들과 충돌했다.

## 무엇이 충돌했는가
- anchor family:
  - support: `8/36` camera-events
  - source: `ippe0` only
  - markers: `id3` x4, `id1` x4
  - raw trace support events: `3, 4, 5, 6`
- cam3 old base solution:
  - method: `cube_anchor`
  - dominant signature: `id3 / ippe1`
  - support: `4/18`
  - events: `9, 10, 11, 12`
- cam3 relative-chain solution:
  - method: `cube_chain_ref`
  - reference: `cam0`
  - support: `14/18`
  - accepted events: `0,1,2,7,8,9,10,11,12,13,14,15,16,17`
  - dominant accepted signature: `id1 / ippe0` (5 events)

즉, anchor는 `event 3~6`의 좁은 family를 기준으로 고정되었고, cam3의 전체 공통 이벤트 분포는 그 family보다 `cam0`과의 상대자세 family를 더 강하게 지지했다.

## 이벤트 레벨 관찰
`event 9~12`에서 cam3 후보는 크게 두 family로 갈렸다.

- anchor-compatible family:
  - `ippe1 / [3]`
  - reprojection: 약 `0.215 px`
  - 현재 old `T_base_C3`와 정확히 일치
- chain-compatible family:
  - `multi / [3,2]`
  - reprojection: 약 `0.256 ~ 0.365 px`
  - old `T_base_C3`와 약 `408 ~ 414 mm`, `47.9 ~ 48.5 deg` 차이
  - 그러나 `cam0` 상대자세와는 일관적

따라서 기존 흔들림은 `cam3 내부 후보가 불안정`해서가 아니라, `anchor-compatible family`와 `relative-chain family` 중 어떤 것을 base 기준으로 채택하느냐의 문제였다.

## 수정
`Step3_calibration.py`에 `evaluate_base_camera_candidate_against_board_refs()`를 추가하고, board로 이미 신뢰되는 카메라(`cam0`, `cam1`) 기준으로 다음 두 후보를 직접 비교하도록 수정했다.

1. `cube_anchor`로 얻은 `T_base_C3`
2. `T_base_C0 @ T_C0_C3`로 얻은 `cube_chain_ref`

선택 기준은 공통 이벤트에서 `T_base_C3 @ T_C_O`가 board-calibrated camera들이 보는 `T_base_O`와 얼마나 일치하는지다.

## 수치 변화
기존 `calib_out_explicit_marker_pose_fixed13`:
- cross-camera mean: `148.19 mm`
- per-camera mean error:
  - cam0: `99.36 mm`
  - cam1: `102.34 mm`
  - cam2: `95.45 mm`
  - cam3: `295.62 mm`

수정 후 `calib_out_cam3_tracefix`:
- cross-camera mean: `18.83 mm`
- per-camera mean error:
  - cam0: `13.33 mm`
  - cam1: `24.34 mm`
  - cam2: `12.44 mm`
  - cam3: `25.22 mm`
- reprojection mean: `1.204 px` (`PASS`)
- hand-eye board stability: `1.21 mm / 0.778 deg` (`PASS`)

## 최종 판단
이제 `cam3`는 더 이상 지배적인 outlier가 아니다. 수정 후 cam3 mean error는 `25.22 mm`로, `cam1 24.34 mm`와 비슷한 수준이다.

남은 오차는 `cam3 단독` 문제가 아니라, cube 전체 marker family가 아직 `5 mm` 이하의 전역 일관성으로 수렴하지 못한 문제다.
