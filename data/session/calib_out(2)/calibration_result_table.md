# Calibration Result Table

## Verification Summary

- Cross-camera: PASS (mean 2.38 mm, median 2.04 mm, max 5.55 mm)
- Cube reprojection: PASS (mean 0.657 px, median 0.636 px, max 1.878 px)
- Hand-eye board stability: PASS (pos std 0.78 mm, rot mean 0.344 deg)
- Board reprojection: PASS (mean 0.324 px)
- Mesh alignment: PASS (mean RMSE 2.89 mm)
- Dimension accuracy: PASS (mean abs err 2.79 mm)
- Pose repeatability: PASS (mean 2.40 mm / 0.529 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 40.617 | 30.711 | -115.724 | 179.891 | -0.426 | 8.206 | 11.080 | 1.406 | 10 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -409.549 | 144.464 | 107.820 | -44.750 | 1.203 | -97.491 | 4.099 | 0.189 | 13 | 14 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 236.654 | 484.543 | 162.757 | 88.099 | -0.267 | -115.725 | 3.526 | 0.419 | 15 | 15 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor_primary+board_refined(a=0.67) | PASS | -249.266 | 911.265 | 123.071 | -158.316 | -0.046 | -109.332 | 2.636 | 0.126 | 12 | 18 | fallback from cube anchor support=18/18 |
| cube | object | T_base_O | cube_consensus_fixed_cams+set_prior_refined | FAIL | -111.268 | 501.506 | 2.379 | -11.614 | -0.173 | 0.274 | 29.617 | 36.574 | 18 | 18 | compatibility average across set anchors |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 221.845 | -151.004 | 687.926 | -172.597 | -46.843 | 142.949 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | PASS | -433.233 | -94.628 | 656.442 | 162.023 | 65.253 | 142.303 | derived from T_base_C0 and T_base_C3 |
