# Cube Failure Report

## Summary

- Cross-camera mean error: 3.31 mm
- Cube reprojection mean error: 0.638 px
- Candidate diagnostics: 212 selected / 212 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 53 | 53 | 1.000 | 4.72 | 0.83 | 7.81 | id2+id3 x12, id4+id3 x9, id2+id1 x9 | id2+id3 x12, id4+id3 x9, id2+id1 x9 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 53 | 53 | 1.000 | 7.91 | 1.24 | 15.04 | id3+id2+id0 x12, id3+id4+id0 x11, id2+id1+id0 x8 | id3+id2+id0 x12, id3+id4+id0 x11, id2+id1+id0 x8 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 53 | 53 | 1.000 | 5.13 | 0.95 |  | id0+id1 x6, id0+id2 x5, id0+id4 x4 | id0+id1 x6, id0+id2 x5, id0+id4 x4 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0+id1 x6, id0+id2 x5, while accepted poses are only id0+id1 x6, id0+id2 x5. |
| cam3 | 53 | 53 | 1.000 | 18.80 | 2.28 | 12.12 | id1+id4+id0 x14, id2+id3 x9, id2+id1 x8 | id1+id4+id0 x14, id2+id3 x9, id2+id1 x8 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 119 | +Z | p0123 | +Z | r0 | 2 | 119 | 2 | 2 | 11.38 | 2.25 | 0.166 | cam2 x2 | current mapping is near-best for this session |
| 1 | 98 | +X | p0123 | -Z | r270 | 55 | 0 | 6 | 6 | 19.37 | 90.04 | 0.209 | cam1 x6 | current face/order ranks poorly (55) against alternatives such as -Z/r270; it never reaches the global inlier threshold |
| 2 | 90 | +Y | p0123 | -Z | r0 | 123 | 89 | 0 | 0 | 15.63 | 1.92 | 0.167 |  | current face/order ranks poorly (123) against alternatives such as -Z/r0; no selected single-marker candidate was accepted |
| 3 | 99 | -X | p0123 | +Z | r90 | 103 | 0 | 0 | 0 | 13.15 | 89.89 | 0.147 |  | current face/order ranks poorly (103) against alternatives such as +Z/r90; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 4 | 99 | -Y | p0123 | +Z | r180 | 173 | 0 | 0 | 0 | 13.74 | 179.26 | 0.225 |  | current face/order ranks poorly (173) against alternatives such as +Z/r180; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
