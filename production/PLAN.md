# Production classifier — plan (pre-registered 2026-11-03, before any production run)

**Goal:** the best possible lens vs. contaminant classifier for the official challenge test set. It is trained on all
200,000 labelled objects.

**Isolation:**
- Everything lives in `production/`: its own configs, experiments and results.
- Shared code is used read-only. The data wrappers are `scripts/data_all.py`, `scripts/train_prod.py` and
  `scripts/run_prod.py`; they run `scripts/train.py` unchanged.
- The frozen `final_kfold_plain` (holdout loss 0.0192) stays the honest reference.

## Data and folds
- **200,000 objects:** train+val (170k) plus the former holdout (30k).
- **Folds:** 10, stratified by simset × 5,000-ID block (`lagn.data.kfold_assign`), giving 20,000 per fold and 5,000 per simset
  per fold.
  - Stored in `results/folds.parquet`; the checks are in `results/fold_checks.json`.
- **Fold model k:** trains on the other 9 folds (180k) and uses fold k only for early stopping and best-epoch choice.
- **Same folds for every family:** for each object, all families' out-of-fold (OOF) predictions come from models that
  never trained on it.

## Recipes (fixed now from earlier validation evidence; no tuning on production OOF)
Preprocessing, augmentation and optimiser are identical to `final_kfold_plain`. Every model is fine-tuned from a
self-supervised encoder. Those encoders were pretrained without labels on the 153k original training stamps only.

| family | config | encoder | why |
|---|---|---|---|
| A ViT | `configs/vit.json` | MAE ViT-B/3px, 400 ep | the ViT half of final_kfold_plain, unchanged |
| B CNN | `configs/cnn800.json` | MIM ConvNeXtV2-B, 800 ep | best single CNN on val (0.0210 vs 0.0217 for e400) |
| C CNN | `configs/cnn1600.json` | MIM ConvNeXtV2-B, 1600 ep | val 0.0208; longer pretraining helped regression a lot |

- Seeds: fold k uses seed k. Fold model = `best.pt`, scored with 8× dihedral TTA.

## Final classifier (rule fixed in advance)
`score = mean over ALL fold models of the family-balanced TTA8 logits`, i.e.:
- logit = ½ · mean(ViT fold models) + ½ · mean(CNN fold models), where CNN = families B and C pooled.
- If family C is not finished by the time the test set arrives, CNN = family B alone.

**No weights, thresholds or member selection are fitted on the OOF predictions.** The one exception is calibration to
the test lens fraction, which is a fixed prior shift and not fitted to labels.

## How it is assessed (no holdout)
1. **OOF score:** every object is predicted by the fold model(s) that never trained on it, using the same
   family-balanced rule with one model per family. Report the log loss, ROC AUC and TPR at FPR 1e-3, overall and per
   pipeline.
   - The OOF score is a slightly *pessimistic* estimate for the 10-model average. Earlier: OOF 0.0198 vs holdout 0.0192.
2. **Bar to beat:**
   - OOF on the old 170k rows ≤ 0.0198 (old OOF).
   - OOF on the 30k former-holdout rows ≤ 0.0192. The old figure is a 20-model average, so this comparison is conservative.
3. **Never-seen check:** compare the OOF loss on the 47k objects the encoders never saw (not even unlabelled) with the
   153k they saw. The official test stamps are also never seen, so the 47k number is the relevant one. If the two
   differ clearly, the pretraining memorised its stamps; report it.
4. **Per-fold spread** (mean ± std of the fold losses), the train vs OOF loss gap per family, and flux-matched / per-flux-
   decile AUC per pipeline (shortcut check).
5. **Integrity:** every run's `val_rows.npy` must equal its fold's rows exactly. `scripts/oof.py` refuses otherwise.

## Known limits (cannot be measured here)
- **Test distribution:** the test set may differ, e.g. a realistic lens fraction or other double/quad and contaminant
  mixes. OOF cannot measure that.
  - A changed lens fraction affects calibration and thresholds, not ranking; recalibrate with the expected prior.
- **Band matching:** test filenames carry only the band, so per-object band matching must be solved before prediction.

## Addendum 2026-11-15 — production version 2: ViT family rebuilt with pad42 (written before any A2 run)
- **Family A2** = family A with exactly one change: the 41×41 stamp gets one zero row and one zero column (pad42,
  14×14 tokens) instead of the bilinear 41→42 resize.
  - Config: `configs/vit_pad42.json`.
  - Encoder: `experiments/ig_pad42/encoder.pt`, made with the same MAE recipe and the same 153k original training
    stamps, so the 47k "never seen" check stays valid.
- **Training:** same 10 folds, fold k uses seed k; run ids `pA2_vitpad42_f{k}`; queue `jobs/queue_pad42.sh`.
- **Basis:** the input-geometry study found no detectable difference (pad42 − interp42 = −4.2 %, −1.45 SE,
  with pad42 lower in 5 of 5 seeds); pad42 keeps the raw pixels.
- **Version 2 classifier:** logit = ½ · mean(A2 fold models) + ½ · mean(B and C fold models).
  - This is identical to version 1 with A replaced by A2. B and C are reused unchanged.
- **Assessment:** OOF exactly as in version 1, on the same 200k rows. Reported next to version 1 and the old model on the
  same rows, plus A2 alone vs A alone paired by fold.
- **Default version, fixed in advance:** version 2, unless its all-200k OOF BCE is
  detectably worse than version 1. "Detectably worse" means delta > +2 SE, using an object bootstrap with 2,000
  resamples.
- **Secondary only, not a selection:** the ViT half pooled over A and A2 (20 models) is reported as a labelled extra row.
