# Cube Failure Report

## Summary

- Cross-camera mean error: 134.21 mm
- Cube reprojection mean error: 0.570 px
- Candidate diagnostics: 48 selected / 0 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 13 | 0 | 0.000 | 75.76 | 46.86 | 405.30 | id4+id1+id0 x6, id2+id3 x4, id1 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id4+id1+id0 x6, id2+id3 x4. |
| cam1 | 9 | 0 | 0.000 | 93.53 | 32.84 | 309.62 | id2+id1 x6, id2+id3 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id2+id1 x6, id2+id3 x3. |
| cam2 | 13 | 0 | 0.000 | 227.97 | 2.11 |  | id0+id3+id2 x3, id0+id2+id3 x2, id0 x1 |  | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0+id3+id2 x3, id0+id2+id3 x2, while accepted poses are only none. |
| cam3 | 13 | 0 | 0.000 | 287.02 | 163.99 | 745.48 | id3+id2 x6, id2 x4, id4+id3+id0 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id3+id2 x6, id2 x4. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 21 | +Z | p0123 | +Z | r0 | 2 | 0 | 1 | 0 | 145.02 | 38.17 | 0.142 | cam2 x1 | current mapping is near-best for this session; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 1 | 18 | +X | p0123 | -X | flip_diag | 81 | 0 | 3 | 0 | 111.75 | 97.98 | 0.092 | cam0 x3 | current face/order ranks poorly (81) against alternatives such as -X/flip_diag; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 2 | 32 | +Y | p0123 | -Y | flip_y | 4 | 0 | 4 | 0 | 203.23 | 37.88 | 0.237 | cam3 x4 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 3 | 23 | -X | p0123 | -X | r90 | 40 | 0 | 0 | 0 | 206.91 | 102.37 | 0.318 |  | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (40) against alternatives such as -X/r90; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 4 | 11 | -Y | p0123 | +X | r270 | 103 | 0 | 0 | 0 | 78.47 | 158.47 | 0.353 |  | current face/order ranks poorly (103) against alternatives such as +X/r270; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
