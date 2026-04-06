# Cube Failure Report

## Summary

- Cross-camera mean error: 6.14 mm
- Cube reprojection mean error: 1.590 px
- Candidate diagnostics: 72 selected / 0 accepted

## Camera-Level Causes

| camera | selected_candidates | accepted_candidates | accept_rate | mean_obj_dt_mm | mean_obj_dr_deg | mean_cam_dt_mm | dominant_selected_markers | dominant_accepted_markers | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0 | 18 | 0 | 0.000 | 167.09 | 58.72 | 506.02 | id3 x7, id4 x5, id2 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id3 x7, id4 x5. |
| cam1 | 18 | 0 | 0.000 | 161.83 | 58.51 | 390.59 | id1 x9, id4 x3, id2 x3 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id1 x9, id4 x3. |
| cam2 | 18 | 0 | 0.000 | 159.84 | 58.66 |  | id0 x12, id2 x4, id4+id0 x1 |  | hand-eye is stable, but cube candidates disagree across markers. Selected poses are dominated by id0 x12, id2 x4, while accepted poses are only none. |
| cam3 | 18 | 0 | 0.000 | 163.17 | 58.71 | 487.74 | id1 x8, id3 x8, id2 x1 |  | camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. Dominant selected markers: id1 x8, id3 x8. |

## Marker-Level Causes

| marker_id | num_observations | current_face | current_perm | best_face | best_perm | current_rank | num_inliers | selected_single | accepted_single | mean_dt_mm | mean_dr_deg | mean_reproj_px | seen_in_cameras | root_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 44 | +Z | p0123 | +Z | flip_anti | 16 | 0 | 13 | 0 | 150.61 | 105.04 | 0.478 | cam2 x12, cam1 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 1 | 28 | +Y | p1032 | -Z | r270 | 98 | 0 | 20 | 0 | 181.86 | 47.54 | 0.305 | cam1 x9, cam3 x8, cam0 x2, cam2 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (98) against alternatives such as -Z/r270; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 2 | 34 | -X | p3210 | +Z | flip_anti | 136 | 0 | 11 | 0 | 169.14 | 98.25 | 0.281 | cam2 x4, cam0 x3, cam1 x3, cam3 x1 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (136) against alternatives such as +Z/flip_anti; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 3 | 29 | -Y | p0321 | -Z | r180 | 158 | 0 | 17 | 0 | 149.14 | 101.32 | 0.173 | cam3 x8, cam0 x7, cam1 x2 | current face/order ranks poorly (158) against alternatives such as -Z/r180; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
| 4 | 28 | +X | p1032 | +Y | r0 | 54 | 0 | 8 | 0 | 168.08 | 111.53 | 0.302 | cam0 x5, cam1 x3 | 2D corner fit is good, but the 3D pose is inconsistent with the global cube model; current face/order ranks poorly (54) against alternatives such as +Y/r0; it never reaches the global inlier threshold; no selected single-marker candidate was accepted |
