# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 8.85 mm, median 7.49 mm, max 23.35 mm)
- Cube reprojection: PASS (mean 1.590 px, median 1.230 px, max 3.691 px)
- Hand-eye board stability: FAIL (pos std 1.08 mm, rot mean 1.093 deg)
- Board reprojection: PASS (mean 0.402 px)
- Mesh alignment: PASS (mean RMSE 3.70 mm)
- Dimension accuracy: PASS (mean abs err 2.90 mm)
- Pose repeatability: FAIL (mean 9.23 mm / 3.181 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | FAIL | 36.270 | 31.827 | -114.541 | 178.952 | -0.311 | 10.331 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -432.454 | 138.525 | 91.896 | -46.596 | -1.963 | -96.151 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 263.419 | 510.985 | 159.610 | 93.294 | -0.853 | -102.038 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_chain_ref | FAIL | -275.525 | 970.619 | 114.033 | -157.771 | -0.307 | -100.567 | 7.365 | 0.750 | 15 | 18 | fallback from cube anchor support=18/18 |
| cube | object | T_base_O | consistent_event_anchor | FAIL | -158.729 | 567.544 | 3.101 | -19.932 | -1.082 | 4.807 | 0.888 | 0.164 | 4 | 18 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 209.761 | -141.809 | 750.627 | -171.815 | -39.500 | 159.274 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | FAIL | -495.653 | -112.391 | 677.638 | 167.153 | 68.279 | 153.417 | derived from T_base_C0 and T_base_C3 |
