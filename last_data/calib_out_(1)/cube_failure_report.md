# Cube Failure Report

## Summary

- Cross-camera mean error: 2.02 mm
- Cube reprojection mean error: 0.602 px
- Candidate diagnostics: 76 selected / 76 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 19 | 19 | 1.000 | 7.47 | 1.15 | 11.34 | id4+id0+id1 x5, id2+id3 x4, id1+id2 x3 | id4+id0+id1 x5, id2+id3 x4, id1+id2 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 19 | 19 | 1.000 | 8.09 | 1.58 | 16.87 | id4+id3+id0 x4, id2 x4, id2+id3 x3 | id4+id3+id0 x4, id2 x4, id2+id3 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 19 | 19 | 1.000 | 4.21 | 0.90 |  | id3+id0+id2 x4, id2+id0 x2, id0+id2+id1 x2 | id3+id0+id2 x4, id2+id0 x2, id0+id2+id1 x2 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id3+id0+id2 x4, id2+id0 x2, while accepted poses are only id3+id0+id2 x4, id2+id0 x2. |
| cam3 | 19 | 19 | 1.000 | 4.40 | 1.05 | 9.28 | id3+id2 x5, id1+id2 x4, id4+id3+id0 x3 | id3+id2 x5, id1+id2 x4, id4+id3+id0 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 38 | +Z | p0123 | +Z | r0 | 2 | 37 | 1 | 1 | 10.31 | 1.82 | 0.201 | cam2 x1 | current mapping is near-best for this session |
| 1 | 30 | +X | p0123 | +Z | r270 | 69 | 0 | 0 | 0 | 11.58 | 90.48 | 0.154 |  | current face/order ranks poorly (69) against alternatives such as +Z/r270; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 2 | 47 | +Y | p0123 | +Z | r0 | 123 | 47 | 4 | 4 | 15.11 | 2.36 | 0.156 | cam1 x4 | current face/order ranks poorly (123) against alternatives such as +Z/r0 |
| 3 | 36 | -X | p0123 | +Z | r90 | 104 | 0 | 0 | 0 | 14.51 | 89.74 | 0.158 |  | current face/order ranks poorly (104) against alternatives such as +Z/r90; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 4 | 26 | -Y | p0123 | +Z | r180 | 172 | 0 | 0 | 0 | 5.88 | 179.44 | 0.170 |  | current face/order ranks poorly (172) against alternatives such as +Z/r180; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
