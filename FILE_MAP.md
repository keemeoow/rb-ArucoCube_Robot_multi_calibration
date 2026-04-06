# File Map

현재 리팩토링 기준의 파일 역할 정리 문서입니다.

## Core Pipeline

- `Step1_dump_all_intrinsics.py`
  - RealSense 카메라 intrinsic 저장
- `Step2_capture_capture.py`
  - 캘리브레이션 세션 캡처
  - RGB, depth, robot pose, board/cube 관측 메타 저장
- `Step3_calibration.py`
  - 메인 캘리브레이션 실행
  - hand-eye, base-to-camera, cube 평균 pose 계산
- `Step4_verify.py`
  - 캘리브레이션 검증
  - cross-camera, reprojection, candidate diagnostics, 시각화
- `Step5_export_reports.py`
  - 결과표, export JSON/NPZ, final-use bundle 생성

## Shared Runtime Modules

- `calibration_runtime_utils.py`
  - Step4/Step5/downstream에서 공통으로 쓰는 런타임 helper
  - cube config 선택, intrinsics 로딩, robot pose 파싱, cube candidate 생성/선택
- `aruco_cube.py`
  - ArUco cube geometry
  - marker detection, pose solving, face-level observation set 생성
- `charuco_utils.py`
  - ChArUco 검출과 pose 추정
- `cube_config_utils.py`
  - cube config JSON/meta 변환
  - 고정 모델/세션 설정 로딩 우선순위 관리
- `config.py`
  - cube, board, dictionary 등 기본 설정
- `downstream_metrics.py`
  - board reprojection, depth/mesh, dimension accuracy, pose repeatability 계산

## Support Modules

- `camera.py`
  - 카메라 접근 helper
- `robot_comm.py`
  - 로봇 포즈 변환 및 통신 helper
- `utils_pose.py`
  - pose averaging, SE(3) distance 등 pose 수학 유틸


## Data Directories

- `data(1)/data/session/`
  - 현재 세션 raw capture 및 남겨둔 결과
- `intrinsics/`
  - 기본 intrinsic 저장 위치
- `configs/cube_models/`
  - 고정 cube model JSON

## Current Run Order

1. `Step1_dump_all_intrinsics.py`
2. `Step2_capture_capture.py`
3. `Step3_calibration.py`
4. `Step4_verify.py`
5. `Step5_export_reports.py`
