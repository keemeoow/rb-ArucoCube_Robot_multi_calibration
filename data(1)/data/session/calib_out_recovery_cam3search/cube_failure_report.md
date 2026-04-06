# Cube Failure Report

## Summary

- Cross-camera mean error: 161.21 mm
- Cube reprojection mean error: 9.033 px
- Candidate diagnostics: 72 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 4 | 0.222 | 269.76 | 58.77 | 419.34 | id2 x9, id4 x7, id1 x2 | id2 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 18 | 4 | 0.222 | 264.86 | 34.40 | 300.51 | id1 x10, id4 x6, id3 x2 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 4 | 0.222 | 261.70 | 66.06 |  | id2 x7, id4 x5, id0 x4 | id2 x2, id4 x1, id1 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id2 x7, id4 x5, while accepted poses are only id2 x2, id4 x1. |
| cam3 | 18 | 4 | 0.222 | 149.03 | 31.72 | 258.63 | id1 x9, id3 x7, id0 x2 | id3 x4 | camera extrinsic is stable, but the dominant selected cube marker (id1 x9) is not the one that survives global checks (id3 x4). This points to marker-to-marker cube model inconsistency. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Y | p1230 | -Z | flip_diag | 146 | 0 | 6 | 0 | 243.30 | 118.33 | 0.488 | cam2 x4, cam3 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (146) against alternatives such as -Z/flip_diag; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 1 | 28 | -Z | p3210 | -Z | flip_y | 67 | 5 | 23 | 5 | 249.92 | 55.42 | 0.346 | cam1 x10, cam3 x9, cam2 x2, cam0 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (67) against alternatives such as -Z/flip_y |
| 2 | 34 | +X | p2301 | +Z | r180 | 69 | 6 | 16 | 6 | 198.04 | 52.17 | 0.276 | cam0 x9, cam2 x7 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (69) against alternatives such as +Z/r180 |
| 3 | 29 | +Z | p2301 | -Z | r180 | 33 | 4 | 9 | 4 | 211.95 | 83.48 | 0.252 | cam3 x7, cam1 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (33) against alternatives such as -Z/r180; only 4 observations reach global consensus |
| 4 | 28 | -X | p2301 | -Z | r180 | 4 | 5 | 18 | 1 | 281.92 | 55.80 | 0.299 | cam0 x7, cam1 x6, cam2 x5 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model |
