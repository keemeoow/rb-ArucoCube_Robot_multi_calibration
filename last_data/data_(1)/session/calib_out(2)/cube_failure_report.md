# Cube Failure Report

## Summary

- Cross-camera mean error: 2.38 mm
- Cube reprojection mean error: 0.657 px
- Candidate diagnostics: 65 selected / 65 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 14 | 14 | 1.000 | 6.27 | 0.84 | 10.67 | id3+id4 x5, id4+id1+id0 x3, id4+id3+id0 x3 | id3+id4 x5, id4+id1+id0 x3, id4+id3+id0 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 15 | 15 | 1.000 | 4.18 | 1.10 | 8.19 | id4+id0 x4, id2+id1+id0 x3, id1+id0+id4 x3 | id4+id0 x4, id2+id1+id0 x3, id1+id0+id4 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 18 | 1.000 | 4.06 | 0.81 |  | id0+id2 x4, id0+id4 x4, id2+id0 x1 | id0+id2 x4, id0+id4 x4, id2+id0 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0+id2 x4, id0+id4 x4, while accepted poses are only id0+id2 x4, id0+id4 x4. |
| cam3 | 18 | 18 | 1.000 | 4.49 | 0.85 | 7.78 | id2+id1+id0 x4, id3+id2 x3, id2+id3 x3 | id2+id1+id0 x4, id3+id2 x3, id2+id3 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

_No rows_
