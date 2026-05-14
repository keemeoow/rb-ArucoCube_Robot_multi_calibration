# Final Use Export

This folder contains the final-use export package for the selected calibration run.

## Included files
- `usable_transforms_final.json`: pass-only transform export with verification snapshot
- `usable_transforms_final.npz`: pass-only 4x4 matrices in NumPy format
- `cube_config_used.json`: cube model/config used for this run
- `calibration_summary_snapshot.json`: full calibration summary snapshot
- `verification_metrics.json`: verification metrics snapshot

## Current quality summary
- Cross-camera mean: 3.3098954807637684
- Reprojection mean: 0.6381875471440064
- Hand-eye pass: False

## Included transforms
- `T_C0_C1` (production)
- `T_C0_C3` (diagnostic)
- `T_base_C0` (production)
- `T_base_C1` (production)
- `T_base_C3` (diagnostic)
