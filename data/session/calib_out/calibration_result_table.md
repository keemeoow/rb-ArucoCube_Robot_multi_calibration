# Calibration Result Table

## Verification Summary

- Cross-camera: PASS (mean 2.34 mm, median 2.12 mm, max 4.69 mm)
- Cube reprojection: PASS (mean 0.657 px, median 0.636 px, max 1.878 px)
- Hand-eye board stability: PASS (pos std 0.73 mm, rot mean 0.339 deg)
- Board reprojection: PASS (mean 0.324 px)
- Mesh alignment: PASS (mean RMSE 2.90 mm)
- Dimension accuracy: PASS (mean abs err 2.75 mm)
- Pose repeatability: PASS (mean 2.33 mm / 0.498 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | FAIL | 40.617 | 30.711 | -115.724 | 179.891 | -0.426 | 8.206 | 11.080 | 1.406 | 10 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | FAIL | -408.890 | 146.657 | 109.664 | -44.917 | 1.264 | -97.670 | 4.187 | 0.217 | 13 | 14 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 239.517 | 483.337 | 160.432 | 87.837 | -0.166 | -115.266 | 1.875 | 0.183 | 11 | 15 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor_primary+board_refined | FAIL | -248.965 | 914.837 | 124.234 | -158.420 | -0.153 | -109.358 | 3.343 | 0.441 | 18 | 18 | fallback from cube anchor support=18/18 |
| cube | object | T_base_O | cube_consensus_fixed_cams+set_prior_refined | FAIL | -110.635 | 502.908 | 2.892 | -11.801 | 0.054 | 0.116 | 30.750 | 35.413 | 18 | 18 | compatibility average across set anchors |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | FAIL | 221.845 | -151.004 | 687.926 | -172.597 | -46.843 | 142.949 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | -433.233 | -94.628 | 656.442 | 162.023 | 65.253 | 142.303 | derived from T_base_C0 and T_base_C3 |
