# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 179.56 mm, median 173.00 mm, max 444.29 mm)
- Cube reprojection: PASS (mean 0.490 px, median 0.464 px, max 1.807 px)
- Hand-eye board stability: PASS (pos std 1.21 mm, rot mean 0.778 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 35.198 | 31.276 | -115.038 | 179.558 | -0.444 | 10.365 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -435.427 | 144.363 | 101.155 | -47.326 | -1.582 | -97.036 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 262.041 | 505.017 | 162.447 | 92.574 | -1.238 | -102.502 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor | FAIL | -594.187 | -62.809 | 125.643 | -46.472 | -2.073 | -101.796 | 5.818 | 0.587 | 4 | 4 | fallback from cube anchor support=4/15 |
| cube | object | T_base_O | board_anchor | FAIL | -351.703 | 353.733 | 62.732 | -14.587 | -48.003 | -88.380 | 9.165 | 1.043 | 8 | 34 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 209.212 | -147.874 | 744.751 | -170.980 | -39.337 | 157.358 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | 45.366 | 8.429 | -258.061 | -0.592 | -0.787 | -4.729 | derived from T_base_C0 and T_base_C3 |
