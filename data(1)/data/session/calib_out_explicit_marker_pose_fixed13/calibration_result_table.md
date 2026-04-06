# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 148.19 mm, median 118.32 mm, max 374.96 mm)
- Cube reprojection: PASS (mean 1.204 px, median 1.251 px, max 3.281 px)
- Hand-eye board stability: PASS (pos std 1.21 mm, rot mean 0.778 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 35.198 | 31.276 | -115.038 | 179.558 | -0.444 | 10.365 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -435.427 | 144.363 | 101.155 | -47.326 | -1.582 | -97.036 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 262.041 | 505.017 | 162.447 | 92.574 | -1.238 | -102.502 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor | FAIL | -293.242 | -166.064 | -153.257 | -8.049 | 4.416 | -69.948 | 0.000 | 0.000 | 4 | 4 | fallback from cube anchor support=4/18 |
| cube | object | T_base_O | board_anchor | FAIL | -283.850 | 326.736 | 1.583 | -15.125 | -2.252 | 12.265 | 4.490 | 0.484 | 8 | 36 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 209.212 | -147.874 | 744.751 | -170.980 | -39.337 | 157.358 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | 317.463 | 274.263 | -72.832 | 1.505 | -39.678 | 25.532 | derived from T_base_C0 and T_base_C3 |
