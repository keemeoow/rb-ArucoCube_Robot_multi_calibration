# Cube Failure Report

## Summary

- Cross-camera mean error: 2.34 mm
- Cube reprojection mean error: 0.657 px
- Candidate diagnostics: 65 selected / 65 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 14 | 14 | 1.000 | 6.32 | 0.88 | 10.98 | id3+id4 x5, id4+id1+id0 x3, id4+id3+id0 x3 | id3+id4 x5, id4+id1+id0 x3, id4+id3+id0 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 15 | 15 | 1.000 | 3.51 | 1.03 | 7.05 | id4+id0 x4, id2+id1+id0 x3, id1+id0+id4 x3 | id4+id0 x4, id2+id1+id0 x3, id1+id0+id4 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 18 | 1.000 | 4.11 | 0.82 |  | id0+id2 x4, id0+id4 x4, id2+id0 x1 | id0+id2 x4, id0+id4 x4, id2+id0 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0+id2 x4, id0+id4 x4, while accepted poses are only id0+id2 x4, id0+id4 x4. |
| cam3 | 18 | 18 | 1.000 | 4.06 | 0.89 | 7.40 | id2+id1+id0 x4, id3+id2 x3, id2+id3 x3 | id2+id1+id0 x4, id3+id2 x3, id2+id3 x3 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 47 | +Z | p1230 | +Y | flip_diag | 6 | 0 | 0 | 0 | 166.27 | 75.91 | 0.241 |  | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 1 | 21 | +Y | p0123 | +Z | flip_y | 117 | 0 | 0 | 0 | 182.93 | 136.33 | 0.394 |  | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (117) against alternatives such as +Z/flip_y; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 2 | 32 | +X | p1230 | +Z | r180 | 73 | 0 | 0 | 0 | 171.87 | 124.94 | 0.536 |  | current face/order ranks poorly (73) against alternatives such as +Z/r180; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 3 | 34 | -Y | p2301 | +Z | r180 | 186 | 0 | 0 | 0 | 163.78 | 115.23 | 0.392 |  | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (186) against alternatives such as +Z/r180; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 4 | 35 | -X | p3012 | +X | r180 | 78 | 0 | 0 | 0 | 162.22 | 131.66 | 0.328 |  | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (78) against alternatives such as +X/r180; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
