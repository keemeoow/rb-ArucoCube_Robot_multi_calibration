# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 134.21 mm, median 135.02 mm, max 138.70 mm)
- Cube reprojection: PASS (mean 0.570 px, median 0.601 px, max 0.894 px)
- Hand-eye board stability: FAIL (pos std 2.37 mm, rot mean 2.151 deg)
- Board reprojection: FAIL (mean 0.502 px)
- Mesh alignment: PASS (mean RMSE 3.05 mm)
- Dimension accuracy: FAIL (mean abs err 6.14 mm)
- Pose repeatability: FAIL (mean 134.21 mm / 39.885 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:ANDREFF | FAIL | 28.532 | 32.760 | -103.052 | 178.537 | 0.148 | 13.583 | 5.583 | 0.316 | 13 | 13 | gripper->camera extrinsic, 13 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | FAIL | 55.724 | 198.514 | 144.817 | 14.564 | -1.894 | -102.467 | 253.435 | 44.346 | 13 | 13 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | FAIL | 412.355 | 312.382 | 89.905 | 61.288 | 3.505 | -95.036 | 0.084 | 0.004 | 4 | 9 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_anchor_primary | FAIL | 17.600 | 353.603 | 119.748 | 3.854 | 8.672 | -98.213 | 221.108 | 68.497 | 13 | 13 | fallback from cube anchor support=13/13 |
| cube | object | T_base_O | cube_consensus_fixed_cams+set_prior_refined | FAIL | -63.077 | 694.870 | -9.144 | -129.737 | 0.131 | 0.575 | 1.842 | 1.109 | 13 | 13 | compatibility average across set anchors |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | FAIL | 142.796 | 28.576 | 1097.763 | 176.081 | -10.515 | -174.063 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | -363.900 | -113.802 | 716.310 | 160.622 | 73.187 | 152.751 | derived from T_base_C0 and T_base_C3 |
