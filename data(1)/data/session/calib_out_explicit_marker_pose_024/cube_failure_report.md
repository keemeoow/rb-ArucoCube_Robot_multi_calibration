# Cube Failure Report

## Summary

- Cross-camera mean error: 179.56 mm
- Cube reprojection mean error: 0.490 px
- Candidate diagnostics: 67 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 16 | 4 | 0.250 | 323.27 | 46.34 | 396.99 | id4 x7, id2 x5, id2+id0 x4 | id2+id0 x4 | camera extrinsic is stable, but the dominant selected cube marker (id4 x7) is not the one that survives global checks (id2+id0 x4). This points to marker-to-marker cube model inconsistency. |
| cam1 | 18 | 4 | 0.222 | 245.91 | 63.20 | 564.68 | id4 x10, id0 x4, id2 x4 | id4 x4 | camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, not from the board-based camera calibration itself. |
| cam2 | 18 | 4 | 0.222 | 289.61 | 58.62 |  | id0 x8, id4 x7, id0+id2 x2 | id0 x2, id0+id2 x1, id4 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x8, id4 x7, while accepted poses are only id0 x2, id0+id2 x1. |
| cam3 | 15 | 4 | 0.267 | 137.63 | 42.32 | 310.68 | id2 x11, id0 x2, id4 x2 | id2 x4 | cam3 extrinsic itself is not board-verified. It depends on cube-anchor only, support 4/15, and accepted cases come from id2 x4. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Y | p0123 | +Z | r180 | 105 | 0 | 14 | 2 | 279.21 | 110.86 | 0.192 | cam2 x8, cam1 x4, cam3 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (105) against alternatives such as +Z/r180; it never reaches the global inlier threshold |
| 2 | 34 | -X | p3210 | +Z | r270 | 139 | 0 | 21 | 4 | 242.91 | 144.56 | 0.446 | cam3 x11, cam0 x5, cam1 x4, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (139) against alternatives such as +Z/r270; it never reaches the global inlier threshold |
| 4 | 28 | +X | p1032 | +Z | r270 | 10 | 0 | 26 | 5 | 329.84 | 74.53 | 0.334 | cam1 x10, cam2 x7, cam0 x7, cam3 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold |
