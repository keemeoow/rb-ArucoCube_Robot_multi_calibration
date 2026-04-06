# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 10.69 mm, median 8.59 mm, max 30.53 mm)
- Cube reprojection: PASS (mean 1.590 px, median 1.230 px, max 3.691 px)
- Hand-eye board stability: PASS (pos std 1.21 mm, rot mean 0.778 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 35.198 | 31.276 | -115.038 | 179.558 | -0.444 | 10.365 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -435.427 | 144.363 | 101.155 | -47.326 | -1.582 | -97.036 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 262.041 | 505.017 | 162.447 | 92.574 | -1.238 | -102.502 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_chain_ref | FAIL | -283.610 | 968.019 | 118.363 | -155.200 | -2.296 | -100.553 | 7.365 | 0.750 | 15 | 18 | fallback from cube anchor support=18/18 |
| cube | object | T_base_O | consistent_event_anchor | FAIL | -274.552 | 332.567 | 3.909 | -15.993 | -2.664 | 10.967 | 2.775 | 0.408 | 4 | 18 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 209.212 | -147.874 | 744.751 | -170.980 | -39.337 | 157.358 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | -502.000 | -112.901 | 661.062 | 167.389 | 71.760 | 153.791 | derived from T_base_C0 and T_base_C3 |
