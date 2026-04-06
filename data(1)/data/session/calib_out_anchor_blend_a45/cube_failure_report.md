# Cube Failure Report

## Summary

- Cross-camera mean error: 10.69 mm
- Cube reprojection mean error: 1.590 px
- Candidate diagnostics: 72 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 4 | 0.222 | 270.94 | 58.58 | 433.48 | id3 x6, id4 x6, id3+id2 x3 | id3 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 18 | 4 | 0.222 | 259.01 | 58.78 | 414.79 | id1 x7, id2 x4, id0 x2 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 4 | 0.222 | 260.98 | 58.31 |  | id0 x10, id3 x2, id2 x2 | id0 x3, id2 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x10, id3 x2, while accepted poses are only id0 x3, id2 x1. |
| cam3 | 18 | 4 | 0.222 | 273.39 | 59.05 | 578.83 | id1 x9, id2 x7, id0 x1 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Z | p0123 | +Z | flip_anti | 16 | 0 | 13 | 3 | 275.18 | 104.28 | 0.478 | cam2 x10, cam1 x2, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold |
| 1 | 28 | +Y | p1032 | +Z | flip_diag | 98 | 9 | 19 | 8 | 226.03 | 51.15 | 0.305 | cam3 x9, cam1 x7, cam0 x2, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (98) against alternatives such as +Z/flip_diag |
| 2 | 34 | -X | p3210 | +Z | flip_anti | 103 | 0 | 14 | 1 | 267.91 | 93.20 | 0.281 | cam3 x7, cam1 x4, cam2 x2, cam0 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (103) against alternatives such as +Z/flip_anti; it never reaches the global inlier threshold |
| 3 | 29 | -Y | p0321 | -Z | r180 | 158 | 0 | 11 | 4 | 308.19 | 103.41 | 0.181 | cam0 x6, cam2 x2, cam1 x2, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (158) against alternatives such as -Z/r180; it never reaches the global inlier threshold |
| 4 | 28 | +X | p1032 | +Y | r0 | 66 | 0 | 10 | 0 | 295.09 | 117.41 | 0.292 | cam0 x6, cam2 x2, cam1 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (66) against alternatives such as +Y/r0; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
