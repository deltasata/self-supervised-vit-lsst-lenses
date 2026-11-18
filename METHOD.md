# Method — lensed AGN vs. contaminant classifier (final: `final_kfold_plain`)

Classification method of the frozen classifier `final_kfold_plain` (2026-10-26). Regression of lens parameters is a
separate phase (`regression/`).

## 1. Data
- **Input:** 5-band (g r i z y) 41×41 stamps, raw pixel values (float32). Models see pixels only. They never see filenames,
  headers, IDs, pipeline flags or truth-table values.
- **Training data:** the challenge training release, 200,000 objects in 4 simsets:
  - SLSim lenses / non-lenses
  - SIMCT lenses / non-lenses
- **Checks before splitting:**
  - no exact or near duplicates (including the 8 rotations/flips)
  - IDs are not paired across simsets
  - no SIMCT background is reused
- **Split:**
  - **Locked holdout:** 30,000 objects (15 %, stratified by simset and 5k-ID block, fixed seed). It was used only for the
    final, frozen models, once each, through a single logged evaluation script.
  - **Development:** the remaining 170,000. The final models use **10-fold cross-validation** over all 170,000; earlier
    models used a fixed 153k/17k train/val split.
  - All model selection, early stopping and ensembling decisions use validation data only.
- **Normalisation:** a fixed per-band transform, the same for every stamp, so colours are preserved: asinh(x / (3 σ_b)),
  where σ_b is the band's median background noise in the training set. It is linear near the noise and logarithmic for
  bright sources, which covers the 10⁵ dynamic range.

## 2. Self-supervised pretraining on the stamps
Both encoders start from public ImageNet self-supervised weights. They then continue self-supervised training on the
153,000 training stamps (unlabelled; holdout never used) and are fine-tuned with labels.

| | ViT branch | CNN branch |
|---|---|---|
| Architecture | ViT-B (85 M params), 3-px patches on a 42×42 input (41→42 bilinear), 196 tokens | ConvNeXtV2-B (88 M params), stride-1 "native" stem at 41×41 |
| ImageNet init | timm `vit_base_patch16_224.mae`; patch embedding re-cut to 3 px × 5 bands | timm `convnextv2_base.fcmae`; first conv inflated to 5 bands |
| Stamp pretraining | MAE, 400 epochs, mask ratio 0.75, light 4-block decoder, normalised-pixel MSE on masked patches | SimMIM-style masked image modelling, 400 epochs, 4-px blocks, mask ratio 0.6, L1 on masked pixels |
| Run | `mae_vitb_p3_42_in1kmae_e400` | `mimF_v2b_e400` |

The 3-px patches and the stride-1 CNN stem preserve the 2.5–10 px image separations; standard 16-px patches would lose
them. ImageNet init followed by in-domain pretraining beat either one alone.

## 3. Supervised fine-tuning (each fold, each branch)
- **Loss:** binary cross-entropy (1 logit). AdamW with weight decay 0.05, layer-wise learning-rate decay 0.75, drop-path
  0.3, EMA of the weights (0.999). Learning rate 1.5e-4 (ViT) and 2e-4 (CNN), with a cosine schedule.
- **Plateau restarts:** when validation stops improving, roll back to the best weights and run a new warm-up/cosine
  cycle with a random peak, up to 8 restarts.
- **Selection:** each epoch is scored on validation loss with 8× rotation/flip test-time augmentation (TTA). The best
  weights are kept.
- **Augmentation:**
  - random rotation/flip (the 8 symmetries of the square)
  - ±1 px integer shift
  - correlated-noise injection (50 % of samples; random amplitude and smoothing). This removes a noise-texture shortcut
    found in the data checks: SIMCT contaminants have more correlated noise.

## 4. Final ensemble
- **`final_kfold_plain`:** 10 folds × 2 branches = 20 models. The score is the mean of the 20 models' TTA8 logits.
- Mixing ViT and CNN branches gave ~9 % lower loss than either alone; a second model of the same family gave ~1 %.
- **Output:** a logit. The probability is trained at the training prior (50/50 lenses), so it should be re-calibrated for
  a realistic lens fraction.

## 5. Results
| | log loss | ROC AUC | AUC SLSim / SIMCT | TPR@FPR 1e-3 SLSim / SIMCT |
|---|---|---|---|---|
| Out-of-fold validation (170,000) | 0.0198 | 0.99975 | — / 0.99925 | — / 0.891 |
| **Locked holdout (30,000)** | **0.0192** | **0.99978** | 0.999984 / 0.99934 | 0.998 / 0.906 |

- **Holdout, more detail:** TPR@FPR 1e-2 is 1.000 on SLSim and 0.982 on SIMCT. PR AUC is 0.99978.
- **Flux-matched holdout AUC:** 0.99956 overall, 0.99855 on SIMCT. The model doesn't depend on the brightness difference
  between lenses and contaminants, which runs in opposite directions in the two pipelines. Its AUC stays at or above
  0.992 in every SIMCT flux decile, lowest in the faintest one.
- **Comparison:** a non-deep baseline on hand-crafted photometric features reaches AUC 0.9838 (SIMCT 0.963).
- **Robust alternative** `final_kfold_robust`: the same recipe plus PSF-width and sky-level augmentation. Holdout loss
  0.0204, AUC 0.99974.

## 6. Compute and reproducibility
- **Hardware:** AMD MI300A GPUs (ROCm), PyTorch 2.9, timm 1.0.
- **Cost:**
  - Fine-tuning per fold model: 10–11 GPU-hours (ViT) and 13–17 GPU-hours (CNN), so about 250 GPU-hours for the
    20-model ensemble.
  - Stamp pretraining: 3.4 GPU-hours (ViT MAE, 400 epochs) and 36 GPU-hours (CNN MIM, 400 epochs).
- **Code:**
  - `scripts/pretrain_mae.py`, `scripts/pretrain_mim_cnn.py`, `scripts/train.py`
  - configs `configs/kfold/{vit_plain,cnn400_plain}.json`
  - ensemble scoring in `scripts/eval_holdout.py`
- **Record:** every run's config and metrics are written to `experiments/<run_id>/`; `scripts/build_registry.py` collects them.
