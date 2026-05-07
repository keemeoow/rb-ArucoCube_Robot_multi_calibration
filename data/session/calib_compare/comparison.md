# Calibration: hand-eye method comparison

- root_folder: `/Users/woo/Documents/GitHub/Robot-Lab/rb-ArucoCube_Robot_multi_calibration/data/session`
- methods evaluated: TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS

Composite score (lower = better) weights downstream object-pose accuracy: cross-cam consistency (most), pose repeatability, hand-eye stability, mesh RMSE.

| Method | Score | cross mean/max (mm) | pose_rep mean/max (mm) | pose_rep rot (°) | HE pos std (mm) | reproj (px) | mesh RMSE (mm) | All Pass |
|---|---|---|---|---|---|---|---|---|
| ANDREFF     | 821.34 | 134.21 / 138.70 | 134.21 / 242.63 | 39.885 |  2.37 | 0.570 |  3.05 | FAIL |
| TSAI        | 822.70 | 134.16 / 138.70 | 134.16 / 242.63 | 39.884 |  3.13 | 0.570 |  3.05 | FAIL |
| DANIILIDIS  | 822.72 | 134.14 / 138.70 | 134.14 / 242.63 | 39.878 |  3.19 | 0.570 |  3.05 | FAIL |
| HORAUD      | 822.73 | 134.15 / 138.70 | 134.15 / 242.63 | 39.885 |  3.16 | 0.570 |  3.05 | FAIL |
| PARK        | 822.75 | 134.15 / 138.70 | 134.15 / 242.63 | 39.886 |  3.16 | 0.570 |  3.05 | FAIL |

**Winner: ANDREFF**  (composite score 821.34)
- cross-cam: mean 134.21mm / max 138.70mm
- pose repeatability: 134.21mm / 39.885°
- mesh alignment RMSE: 3.05mm