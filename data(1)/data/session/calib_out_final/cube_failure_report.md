# Cube Failure Report

## Summary

- Cross-camera mean error: 163.17 mm
- Cube reprojection mean error: 12.746 px
- Candidate diagnostics: 72 selected / 16 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 4 | 0.222 | 259.05 | 75.93 | 502.22 | id3 x8, id0 x4, id4 x4 | id0 x4 | camera extrinsic is stable, but the dominant selected cube marker (id3 x8) is not the one that survives global checks (id0 x4). This points to marker-to-marker cube model inconsistency. |
| cam1 | 18 | 4 | 0.222 | 254.18 | 39.94 | 347.18 | id1 x5, id4+id1 x4, id0 x4 | id4+id1 x4 | camera extrinsic is stable, but the dominant selected cube marker (id1 x5) is not the one that survives global checks (id4+id1 x4). This points to marker-to-marker cube model inconsistency. |
| cam2 | 18 | 4 | 0.222 | 248.59 | 53.70 |  | id0 x10, id4 x4, id0+id1 x1 | id0 x2, id4 x1, id0+id1 x1 | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x10, id4 x4, while accepted poses are only id0 x2, id4 x1. |
| cam3 | 18 | 4 | 0.222 | 153.92 | 57.52 | 424.54 | id3 x7, id1 x6, id2 x3 | id3 x4 | cam3 extrinsic itself is not board-verified. It depends on cube-anchor only, support 4/18, and accepted cases come from id3 x4. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | -Y | p1032 | +Z | r90 | 163 | 8 | 19 | 6 | 228.69 | 66.25 | 0.165 | cam2 x10, cam0 x4, cam1 x4, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (163) against alternatives such as +Z/r90 |
| 1 | 28 | +X | p1032 | +Z | r0 | 56 | 5 | 14 | 0 | 243.06 | 76.71 | 0.346 | cam3 x6, cam1 x5, cam0 x2, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (56) against alternatives such as +Z/r0; no selected single-marker candidate was accepted |
| 2 | 34 | -X | p1230 | +Z | r0 | 118 | 0 | 4 | 0 | 192.18 | 114.70 | 0.304 | cam3 x3, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (118) against alternatives such as +Z/r0; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 3 | 29 | +Z | p0123 | +Z | r0 | 2 | 4 | 17 | 4 | 202.27 | 62.91 | 0.179 | cam0 x8, cam3 x7, cam1 x2 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current mapping is near-best for this session; only 4 observations reach global consensus |
| 4 | 28 | -Z | p1032 | +X | r0 | 8 | 5 | 12 | 1 | 275.65 | 51.77 | 0.287 | cam2 x4, cam0 x4, cam1 x3, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model |
