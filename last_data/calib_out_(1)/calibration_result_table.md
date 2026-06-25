# Calibration Result Table

## Verification Summary

- Cross-camera: PASS (mean 2.02 mm, median 1.88 mm, max 3.94 mm)
- Cube reprojection: PASS (mean 0.602 px, median 0.621 px, max 1.059 px)
- Hand-eye board stability: PASS (pos std 0.46 mm, rot mean 0.650 deg)
- Board reprojection: PASS (mean 0.430 px)
- Mesh alignment: PASS (mean RMSE 2.50 mm)
- Dimension accuracy: PASS (mean abs err 1.87 mm)
- Pose repeatability: PASS (mean 2.02 mm / 0.517 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:DANIILIDIS | PASS | 32.144 | 31.734 | -113.653 | 179.510 | -0.436 | 12.852 | 6.456 | 0.824 | 18 | 19 | gripper->camera extrinsic, 19 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | FAIL | -404.767 | 150.876 | 107.930 | -42.517 | -0.467 | -101.499 | 6.661 | 0.205 | 11 | 19 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | FAIL | 352.418 | 569.542 | 100.183 | 93.095 | -1.925 | -105.488 | 2.308 | 0.044 | 9 | 19 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor_primary+board_refined(a=0.45) | PASS | -248.582 | 960.001 | 119.289 | -159.166 | 3.527 | -99.954 | 4.313 | 0.387 | 14 | 19 | fallback from cube anchor support=19/19 |
| cube | object | T_base_O | cube_consensus_fixed_cams+set_prior_refined | FAIL | -108.240 | 462.216 | 0.431 | 174.096 | -0.296 | 0.545 | 39.885 | 47.237 | 19 | 19 | compatibility average across set anchors |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | FAIL | 283.964 | -150.021 | 807.685 | -164.433 | -43.213 | 145.909 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | -428.744 | -160.916 | 688.774 | 153.032 | 58.938 | 139.867 | derived from T_base_C0 and T_base_C3 |
