# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 163.17 mm, median 153.44 mm, max 225.12 mm)
- Cube reprojection: FAIL (mean 12.740 px, median 9.139 px, max 61.027 px)
- Hand-eye board stability: PASS (pos std 1.21 mm, rot mean 0.778 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 35.198 | 31.276 | -115.038 | 179.558 | -0.444 | 10.365 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -435.427 | 144.363 | 101.155 | -47.326 | -1.582 | -97.036 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 262.041 | 505.017 | 162.447 | 92.574 | -1.238 | -102.502 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor_strict | PASS | 87.756 | 27.883 | 135.792 | 40.884 | 1.085 | -102.359 | 0.000 | 0.000 | 4 | 4 | fallback from cube anchor support=4/18 |
| cube | object | T_base_O | board_anchor | FAIL | -266.739 | 335.483 | 1.202 | 158.788 | 0.554 | -89.011 | 6.950 | 0.823 | 8 | 36 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 209.212 | -147.874 | 744.751 | -170.980 | -39.337 | 157.358 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | PASS | 441.054 | -59.745 | 300.649 | -73.367 | -83.843 | 62.715 | derived from T_base_C0 and T_base_C3 |
