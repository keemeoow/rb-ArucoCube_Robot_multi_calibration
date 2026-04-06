# Cube Failure Report

## Summary

- Cross-camera mean error: 12.15 mm
- Cube reprojection mean error: 4.820 px
- Candidate diagnostics: 72 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 4 | 0.222 | 265.59 | 58.08 | 430.39 | id4 x6, id2 x4, id3 x4 | id3 x2, id0 x2 | camera extrinsic is stable, but the dominant selected cube marker (id4 x6) is not the one that survives global checks (id3 x2). This points to marker-to-marker cube model inconsistency. |
| cam1 | 18 | 4 | 0.222 | 255.09 | 58.52 | 407.45 | id1 x5, id4 x4, id2 x4 | id1 x3, id4+id1 x1 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 4 | 0.222 | 256.79 | 58.36 |  | id0 x12, id2 x2, id4 x2 | id0 x2, id2 x1, id4 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x12, id2 x2, while accepted poses are only id0 x2, id2 x1. |
| cam3 | 18 | 4 | 0.222 | 273.64 | 58.35 | 571.01 | id1 x8, id3 x5, id2 x3 | id1 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Z | p0123 | +Z | flip_anti | 17 | 0 | 16 | 4 | 270.50 | 104.27 | 0.478 | cam2 x12, cam0 x2, cam1 x1, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold |
| 1 | 28 | +Y | p1032 | -Z | r270 | 98 | 9 | 16 | 7 | 224.92 | 49.55 | 0.305 | cam3 x8, cam1 x5, cam0 x2, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (98) against alternatives such as -Z/r270 |
| 2 | 34 | -X | p3210 | +Z | flip_anti | 103 | 0 | 13 | 1 | 264.95 | 95.29 | 0.281 | cam0 x4, cam1 x4, cam3 x3, cam2 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (103) against alternatives such as +Z/flip_anti; it never reaches the global inlier threshold |
| 3 | 29 | -Y | p0321 | +Y | r0 | 158 | 0 | 13 | 2 | 304.05 | 103.22 | 0.181 | cam3 x5, cam0 x4, cam1 x3, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (158) against alternatives such as +Y/r0; it never reaches the global inlier threshold |
| 4 | 28 | +X | p1032 | +Y | r0 | 62 | 0 | 13 | 1 | 290.89 | 116.63 | 0.292 | cam0 x6, cam1 x4, cam2 x2, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (62) against alternatives such as +Y/r0; it never reaches the global inlier threshold |
