# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 12.15 mm, median 10.66 mm, max 31.63 mm)
- Cube reprojection: FAIL (mean 4.820 px, median 3.015 px, max 17.534 px)
- Hand-eye board stability: PASS (pos std 1.21 mm, rot mean 0.778 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 35.198 | 31.276 | -115.038 | 179.558 | -0.444 | 10.365 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -435.427 | 144.363 | 101.155 | -47.326 | -1.582 | -97.036 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 262.041 | 505.017 | 162.447 | 92.574 | -1.238 | -102.502 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_chain_ref | FAIL | -304.674 | 962.908 | 121.831 | -153.515 | 3.104 | -101.193 | 27.191 | 3.486 | 15 | 18 | fallback from cube anchor support=18/18 |
| cube | object | T_base_O | consistent_event_anchor | FAIL | -268.593 | 337.141 | 7.038 | -16.478 | -2.740 | 9.828 | 0.106 | 0.013 | 4 | 18 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 209.212 | -147.874 | 744.751 | -170.980 | -39.337 | 157.358 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | -512.420 | -114.309 | 641.794 | 149.690 | 71.090 | 136.349 | derived from T_base_C0 and T_base_C3 |
