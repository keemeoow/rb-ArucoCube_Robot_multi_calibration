# Calibration Mode Comparison

Planar board seed, cube-only, and hybrid refinement were re-evaluated on the same dataset.

| mode | num_base_cameras | base_cameras | cross_camera_mean_mm | cube_reproj_mean_px | board_reproj_mean_px | mesh_rmse_mm | dimension_err_mm | pose_repeat_mm | pose_repeat_deg | handeye_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| board_only | 2 | cam0, cam1 | 8.98 | 1.590 | 0.402 | 3.66 | 3.05 | 9.21 | 2.236 | PASS |
| cube_only | 3 | cam0, cam1, cam3 | 7.25 | 1.590 | 0.402 | 3.66 | 2.75 | 7.48 | 3.104 | FAIL |
| hybrid | 3 | cam0, cam1, cam3 | 6.14 | 1.590 | 0.402 | 3.70 | 4.05 | 6.22 | 2.034 | PASS |
