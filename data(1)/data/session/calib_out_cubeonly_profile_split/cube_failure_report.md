# Cube Failure Report

## Summary

- Cross-camera mean error: 6.14 mm
- Cube reprojection mean error: 1.590 px
- Candidate diagnostics: 72 selected / 0 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 0 | 0.000 | 167.09 | 58.72 | 506.02 | id3 x7, id4 x5, id2 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id3 x7, id4 x5. |
| cam1 | 18 | 0 | 0.000 | 161.83 | 58.51 | 390.59 | id1 x9, id4 x3, id2 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id1 x9, id4 x3. |
| cam2 | 18 | 0 | 0.000 | 159.84 | 58.66 |  | id0 x12, id2 x4, id4+id0 x1 |  | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x12, id2 x4, while accepted poses are only none. |
| cam3 | 18 | 0 | 0.000 | 163.17 | 58.71 | 487.74 | id1 x8, id3 x8, id2 x1 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id1 x8, id3 x8. |

## Marker-Level Causes

_No rows_
