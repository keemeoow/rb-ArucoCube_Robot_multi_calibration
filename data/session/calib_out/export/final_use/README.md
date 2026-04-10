# Final Use Export

This folder contains the final-use export package for the selected calibration run.

## Included files
- `usable_transforms_final.json`: pass-only transform export with verification snapshot
- `usable_transforms_final.npz`: pass-only 4x4 matrices in NumPy format
- `cube_config_used.json`: cube model/config used for this run
- `calibration_summary_snapshot.json`: full calibration summary snapshot
- `verification_metrics.json`: verification metrics snapshot

## Current quality summary
- Cross-camera mean: 2.335451276500878
- Reprojection mean: 0.656836755290816
- Hand-eye pass: True

## Included transforms
- `T_base_C1` (production)
