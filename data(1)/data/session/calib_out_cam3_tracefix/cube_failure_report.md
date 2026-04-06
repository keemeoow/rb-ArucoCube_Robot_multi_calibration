# Cube Failure Report

## Summary

- Cross-camera mean error: 13.22 mm
- Cube reprojection mean error: 1.204 px
- Candidate diagnostics: 72 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 4 | 0.222 | 283.31 | 58.82 | 440.06 | id3 x6, id3+id2 x5, id4 x5 | id3 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 18 | 4 | 0.222 | 265.18 | 58.58 | 418.63 | id1 x9, id4 x4, id0 x2 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 4 | 0.222 | 271.59 | 58.56 |  | id0 x9, id4 x4, id0+id2 x3 | id0+id2 x2, id4 x1, id0 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x9, id4 x4, while accepted poses are only id0+id2 x2, id4 x1. |
| cam3 | 18 | 4 | 0.222 | 291.64 | 60.16 | 596.81 | id1 x9, id2 x7, id0 x1 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Z | p0123 | +Z | flip_anti | 16 | 0 | 12 | 1 | 286.62 | 104.40 | 0.478 | cam2 x9, cam1 x2, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold |
| 1 | 28 | +Y | p1032 | +Z | flip_diag | 98 | 5 | 21 | 8 | 236.06 | 52.73 | 0.305 | cam3 x9, cam1 x9, cam0 x2, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (98) against alternatives such as +Z/flip_diag |
| 2 | 34 | -X | p3210 | +Z | flip_anti | 103 | 0 | 9 | 0 | 280.37 | 91.32 | 0.281 | cam3 x7, cam2 x1, cam1 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (103) against alternatives such as +Z/flip_anti; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 3 | 29 | -Y | p0321 | -Z | r180 | 158 | 0 | 8 | 4 | 321.31 | 102.85 | 0.181 | cam0 x6, cam1 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (158) against alternatives such as -Z/r180; it never reaches the global inlier threshold |
| 4 | 28 | +X | p1032 | +Y | r0 | 66 | 0 | 14 | 1 | 302.86 | 118.48 | 0.292 | cam0 x5, cam1 x4, cam2 x4, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (66) against alternatives such as +Y/r0; it never reaches the global inlier threshold |
