# cam1 / marker-family Trace Report

## 결론
`cam1`의 남은 오차는 `T_base_C1`이 흔들려서가 아니라, 프레임마다 cube 후보를 고를 때 `multi-marker` family를 과신하던 선택 규칙 때문에 생겼다.

`T_base_C1` 자체는 board 기반으로 안정적이다.
- board-based translation std: `1.44 mm`
- board-based rotation std: `0.221 deg`

하지만 cube 후보 선택에서는 기존 primary 규칙이 `marker 수가 많은 multi-marker 후보`를 항상 먼저 고르고 있었고, 실제 base 일관성 기준으로는 `single-marker ippe0` 후보가 더 맞았다.

## 핵심 관찰
cam1의 18개 이벤트 모두에서 naive primary와 consistency-aware 선택이 달랐다.

- naive primary: `18/18` events 모두 `multi`
- refined selection: `18/18` events 모두 `ippe0`

기존 primary signature 분포:
- `[4,1] / multi`: 4
- `[1,2,0] / multi`: 4
- `[1,0,4] / multi`: 3
- `[4,3,0] / multi`: 2
- `[1,4,0] / multi`: 2
- `[4,0] / multi`: 1
- `[2,3,0] / multi`: 1
- `[2,0,3] / multi`: 1

refined signature 분포:
- `[1] / ippe0`: 9
- `[4] / ippe0`: 4
- `[0] / ippe0`: 2
- `[3] / ippe0`: 2
- `[2] / ippe0`: 1

## 이벤트별 패턴
대표 예시:
- event 3~6:
  - primary: `[4,1] / multi`, reproj 약 `1.30 ~ 1.37 px`
  - refined: `[1] / ippe0`, reproj 약 `0.05 ~ 0.23 px`
- event 9~12:
  - primary: `[1,2,0] / multi`, reproj 약 `1.55 ~ 1.70 px`
  - refined: `[1] / ippe0` 또는 `[2] / ippe0`, reproj 약 `0.08 ~ 0.16 px`
- event 16~17:
  - primary: `[2,3,0] / multi` 또는 `[2,0,3] / multi`, reproj 약 `1.71 px`
  - refined: `[3] / ippe0`, reproj 약 `0.007 px`

즉 cam1에서는 “marker를 많이 쓴 후보”가 꼭 더 좋은 cube pose가 아니었다. 오히려 현재 cube 모델에서는 multi-marker family가 서로 다른 marker basis를 섞으면서 base 일관성을 깨고 있었다.

## 수치 변화
기존 `cam3_tracefix`에서 naive primary selection 사용 시:
- cross-camera mean: `18.83 mm`
- cam1 mean error: 약 `24.34 mm`

consistency-aware refinement 적용 후:
- cross-camera mean: `13.22 mm`
- cam1 mean error: 약 `16.62 mm`

같은 refinement 기준에서 카메라별 mean error:
- cam0: `9.44 mm`
- cam1: `16.62 mm`
- cam2: `10.12 mm`
- cam3: `16.71 mm`

## 최종 판단
- `cam1` 문제의 본질은 board extrinsic이 아니라 `cube candidate family selection`이다.
- 남은 오차는 `cam1`과 `cam3` 둘 다에서 비슷한 수준으로 남고 있다.
- 따라서 이제는 특정 카메라 하나의 extrinsic보다는, marker family 전체가 아직 완전한 rigid cube model로 수렴하지 않는 문제가 주된 잔차다.
