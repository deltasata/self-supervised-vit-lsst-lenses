# Regression plan (2026-10-26)

## Decisions
- Targets (common to SLSim D1_L and SIMCT D2_L):
  - core: Einstein radius, double/quad, mass ellipticity (q, PA as e1/e2 components), source position relative to the
    lens, lens redshift;
  - secondary: source redshift, external shear.
  - Skip: vel_disp (derived), magnification (poorly constrained), SLSim-only image positions / time delays.
- Lenses only: regression loss on lens rows; non-lens rows carry no targets (D2_N lens-model columns are not physical).
  Each output gets a predicted uncertainty (Gaussian NLL).
- Separate models first: the frozen final_kfold_plain stays the classifier; regression models start from the same
  pretrained encoders (400-ep stamp MAE ViT-B, 400-ep stamp MIM ConvNeXtV2-B). A joint classification+regression model
  is a later comparison.
- Same locked holdout; frozen-before-evaluation; one evaluation each.

## Target conventions (verified 2026-10-26)
Pixel frame: x → +column, y → +row, 0.2″/px, lens at pixel (20, 20). Angles below are measured from +x towards +y.
| quantity | SLSim D1_L | SIMCT D2_L |
|---|---|---|
| source position | srcx/srcy (= psx/psy), arcsec | source_x/source_y in **HSC pixels** (×0.168 → arcsec) |
| mass axis ratio q | ell_m = (1−q)/(1+q) | ell_m = 1−q (probable; see caveat) |
| mass major-axis angle | 90° − ell_m_PA | −ell_m_PA |
| shear angle (γ1 = γ cos 2θ) | 90° − sh_PA | not testable (γ ≤ 0.02); assume −sh_PA like the mass PA |
| Einstein radius | Rein, arcsec | Rein, arcsec |
Caveats: SLSim reproduces tabulated images only to ~0.07″ with an SIE (likely a power-law slope that is not tabulated).
D2 ellipticity: "1−q" gives 100 % of doubles but 77 % of quads; "(1−q)/(1+q)" 85 % / 84 %; the 1−q choice is backed by the
q distributions (D1 median q 0.71, D2 0.68 vs 0.52 under the other definition). Ask the organizers to confirm.
The D2 Einstein-radius normalisation (which SIE definition) is unconfirmed.

## Target set used
Only quantities common to both pipelines whose convention is settled:
theta_e, src_x, src_y (arcsec, lens-centred), z_lens, z_src (Gaussian NLL, mean + log-variance each) and is_quad (BCE).
Excluded: mass ellipticity and orientation (option a; D2 axis-ratio definition only probable), shear (D2 angle
untestable, D2 γ ≤ 0.02), vel_disp (derived from θE and redshifts), magnification and source/AGN magnitudes (definitions
differ between pipelines). Table: regression/scripts/targets.py -> regression/results/targets_trainval.parquet.

## Next steps (original list; 1-3 done 2026-10-26)
1. Target table builder (regression/scripts/targets.py): one row per lens, converted to the common frame above
   (theta_E, q, e1/e2 of the mass, source x/y in arcsec relative to the lens, zl, zs, gamma1/gamma2, is_quad).
2. Train/val split = the classification split (holdout untouched); k-fold later if useful.
3. Baselines: ViT-B 400-ep MAE encoder and ConvNeXtV2-B 400-ep MIM encoder with a regression head, lenses only.
