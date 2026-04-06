# Calibration Result Table

## Verification Summary

- Cross-camera: FAIL (mean 6.10 mm, median 5.27 mm, max 22.85 mm)
- Cube reprojection: PASS (mean 1.590 px, median 1.230 px, max 3.691 px)
- Hand-eye board stability: PASS (pos std 1.59 mm, rot mean 0.805 deg)
- Board reprojection: PASS (mean 0.402 px)
- Mesh alignment: PASS (mean RMSE 3.73 mm)
- Dimension accuracy: FAIL (mean abs err 3.83 mm)
- Pose repeatability: FAIL (mean 6.16 mm / 2.057 deg)

## Camera / Object Transforms

| entity | role | transform | solve_method | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | error_trans_mm | error_rot_deg | support_inliers | support_total | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam2 | gripper | T_gripper_cam | hand-eye:HORAUD | PASS | 33.466 | 29.918 | -117.784 | 179.305 | -0.724 | 10.353 | 1.914 | 0.394 | 18 | 18 | gripper->camera extrinsic, 18 ChArUco frames |
| cam0 | fixed | T_base_C0 | board-based | PASS | -445.877 | 140.403 | 91.969 | -48.126 | -1.893 | -96.053 | 1.594 | 0.268 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam1 | fixed | T_base_C1 | board-based | PASS | 268.530 | 508.799 | 161.050 | 92.944 | -0.587 | -102.126 | 1.444 | 0.221 | 17 | 17 | fixed camera extrinsic from ChArUco board |
| cam3 | fixed | T_base_C3 | cube_chain_ref | PASS | -280.346 | 975.760 | 112.828 | -157.974 | 0.237 | -100.097 | 7.365 | 0.750 | 15 | 18 | fallback from cube anchor support=18/18 |
| cube | object | T_base_O | consistent_event_anchor | FAIL | -152.892 | 557.726 | 4.296 | -24.503 | -1.116 | 6.513 | 2.775 | 0.408 | 4 | 18 | static cube anchor in robot base |

## Relative Transforms

| transform | status | x_mm | y_mm | z_mm | rz_deg | ry_deg | rx_deg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T_C0_C1 | PASS | 204.725 | -144.025 | 766.948 | -172.523 | -38.393 | 159.676 | derived from T_base_C0 and T_base_C1 |
| T_C0_C3 | PASS | -510.556 | -109.321 | 673.081 | 164.817 | 69.405 | 151.809 | derived from T_base_C0 and T_base_C3 |
