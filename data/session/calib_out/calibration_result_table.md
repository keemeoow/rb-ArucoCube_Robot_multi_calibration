# Calibration Result Table

## Verification Summary

- Cross-camera: PASS (mean 3.31 mm, median 2.51 mm, max 9.95 mm)
- Cube reprojection: PASS (mean 0.638 px, median 0.616 px, max 1.634 px)
- Hand-eye board stability: FAIL (pos std 2.15 mm, rot mean 1.366 deg)
- Board reprojection: PASS (mean 0.402 px)
- Mesh alignment: PASS (mean RMSE 2.67 mm)
- Dimension accuracy: PASS (mean abs err 2.78 mm)
- Pose repeatability: PASS (mean 3.32 mm / 0.690 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:TSAI | FAIL | 34.944 | 30.772 | -115.010 | 178.638 | -1.792 | 8.034 | 3.908 | 0.759 | 53 | 53 | gripper->camera extrinsic, 53 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -430.358 | 109.758 | 106.319 | -34.301 | -2.816 | -99.835 | 2.531 | 0.259 | 44 | 48 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 355.125 | 513.386 | 177.629 | 92.344 | 1.755 | -101.852 | 2.529 | 0.272 | 49 | 51 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | board_based+board_refined(a=0.70) | PASS | -296.074 | 930.070 | 121.185 | -161.012 | 4.901 | -102.562 | 1.944 | 0.283 | 44 | 52 | fallback from cube anchor support=44/52 |
| cube | object | T_base_O | cube_consensus_fixed_cams | FAIL | -148.566 | 467.567 | 0.357 | 154.359 | 0.256 | 0.333 | 36.532 | 52.987 | 53 | 53 | compatibility average across set anchors |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 425.447 | -182.532 | 764.671 | -167.663 | -52.199 | 154.852 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | PASS | -353.385 | -162.606 | 742.553 | 163.210 | 49.336 | 146.296 | derived from T_base_C0 and T_base_C3 |
