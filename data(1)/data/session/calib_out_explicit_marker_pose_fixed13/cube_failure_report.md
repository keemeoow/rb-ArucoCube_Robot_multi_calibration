# Cube Failure Report

## Summary

- Cross-camera mean error: 148.19 mm
- Cube reprojection mean error: 1.204 px
- Candidate diagnostics: 72 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 4 | 0.222 | 285.30 | 52.84 | 424.05 | id3 x10, id4 x4, id1 x2 | id3 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam1 | 18 | 4 | 0.222 | 264.79 | 37.28 | 335.24 | id1 x10, id4 x3, id0 x3 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 4 | 0.222 | 266.12 | 55.38 |  | id0 x4, id2 x4, id4 x4 | id2 x2, id0 x1, id1 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x4, id2 x4, while accepted poses are only id2 x2, id0 x1. |
| cam3 | 18 | 4 | 0.222 | 164.91 | 55.99 | 470.29 | id3 x7, id2 x5, id1 x4 | id3 x4 | cam3 extrinsic itself is not board-verified. It depends on cube-anchor only, support 4/18, and accepted cases come from id3 x4. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Z | p0123 | +Z | flip_anti | 11 | 0 | 7 | 1 | 246.05 | 102.96 | 0.478 | cam2 x4, cam1 x3 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold |
| 1 | 28 | +Y | p1032 | +Z | flip_diag | 133 | 5 | 19 | 5 | 251.06 | 92.16 | 0.345 | cam1 x10, cam3 x4, cam2 x3, cam0 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (133) against alternatives such as +Z/flip_diag |
| 2 | 34 | -X | p3210 | -Z | r180 | 137 | 0 | 10 | 2 | 203.61 | 97.25 | 0.285 | cam3 x5, cam2 x4, cam0 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (137) against alternatives such as -Z/r180; it never reaches the global inlier threshold |
| 3 | 29 | -Y | p0321 | -Z | r180 | 156 | 0 | 21 | 8 | 223.64 | 95.15 | 0.189 | cam0 x10, cam3 x7, cam2 x2, cam1 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (156) against alternatives such as -Z/r180; it never reaches the global inlier threshold |
| 4 | 28 | +X | p1032 | +Y | r0 | 62 | 0 | 13 | 0 | 290.52 | 118.38 | 0.270 | cam0 x4, cam2 x4, cam1 x3, cam3 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (62) against alternatives such as +Y/r0; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
