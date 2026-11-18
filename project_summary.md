# Project summary: finding strongly lensed AGN in simulated LSST images with deep learning

*LSST Lensed AGN Static Search Challenge (agnchallenge.cbpf.br), 2026-10-06 → 2026-11-18. Status as of 2026-11-18.*

Every number below comes from the project's run records and lab notebook.
Each table states the model, the objects it was scored on, and the metric.

- **BCE** = binary cross-entropy (log loss) of the lens-vs-contaminant prediction. Lower is better.
- **AUC** = ROC area under the curve.
- **TPR@FPR 1e-3** = the fraction of lenses found when only 1 in 1,000 contaminants passes.
- **SLSim** (pipeline D1) = fully synthetic images.
- **SIMCT** (pipeline D2) = real HSC images with injected sources, degraded to LSST quality; the better proxy for real data.

---

## 1. The problem

Strongly lensed quasars (AGN) are rare: roughly one in thousands of candidate objects. The task is to separate them
from look-alikes such as quasar pairs, stars next to galaxies, and compact galaxies. The images are LSST first-year
co-adds.

- **Input:** 5-band (g r i z y) stamps of 41×41 pixels (0.2″/px), raw flux values.
- **Training data:** 200,000 labelled objects in 4 balanced sets (lenses / contaminants × SLSim / SIMCT).
- **What makes it hard:**
  - Lensed-image separations are only 2.5–10 px, against a PSF of about 4 px.
  - Brightness spans a dynamic range of 10⁵.
  - The two pipelines have **opposite brightness shortcuts**: SLSim lenses are brighter than SLSim contaminants, while
    SIMCT lenses are fainter.
- **Later in the project**, truth tables (lens-model parameters) arrived. They were used for:
  - error analysis,
  - a lens-parameter **regression** model,
  - auxiliary training targets (multi-task learning).

## 2. Data verification and experimental protocol

**Checks on all 1,000,000 FITS files** (scripted, recorded with exact numbers):
- counts, formats and NaNs
- exact duplicates (hashes) and near duplicates, including under all 8 rotations/flips, with a test that is robust to
  the bright central source
- ID pairing across sets
- reuse of SIMCT backgrounds
- per-band noise levels
- empirical PSF widths (FWHM ≈ 3.7–6.1 px)
- truth-table coordinate and photometry conventions (zero point 27.0 for SLSim)

**Shortcut candidates found in the data**, each later tested explicitly:
- reversed brightness between the two pipelines
- noise texture: SIMCT contaminants have more spatially correlated noise
- centroid offsets
- the outer-background level

**Protocol:**
- **Locked holdout:** 30,000 objects (15 %, stratified), write-protected and verified by SHA-256 hashes. Only one logged
  script can read it, and it refuses models that are not frozen.
- **Selection data:** all model selection, early stopping and ensembling used validation data or out-of-fold
  predictions, never the holdout.
- **Inputs:** pixels only. Filenames, headers, IDs and pipeline flags are never inputs.
- **Pre-registered comparisons:** for close calls, the analysis and the decision rule were written down before the
  results were seen (Section 7).

**Baseline 0, a hand-crafted floor:** a classifier on per-band aperture fluxes, peaks, concentration, colours and sizes.
Every deep model had to beat it clearly. Scored on the 17,000 validation objects:

| baseline model | val AUC | val BCE | AUC SLSim | AUC SIMCT |
|---|---|---|---|---|
| logistic regression | 0.9077 | 0.388 | 0.934 | 0.888 |
| MLP (256, 256) | **0.9838** | 0.156 | 0.996 | 0.963 |
| LightGBM | 0.9812 | 0.165 | 0.995 | 0.960 |

## 3. Models tested (about 300 classification runs, 40 production fold models, about 50 regression runs)

**Common setup:**
- Every model sees the 5 bands through a fixed per-band asinh stretch, the same for every stamp, so colours are kept.
- Scores are on the same 17,000 validation objects, using the best checkpoint with 8× rotation/flip test-time
  averaging (TTA8).

**Caution on comparing rows:** early screening runs used a short recipe (about 30 epochs). Later runs used the full
recipe: layer-wise learning-rate decay, restarts, EMA. Differences below about 0.001 BCE between single runs are within
seed noise, as measured by a paired bootstrap.

### 3.1 Best validation result per architecture (selected rows from the run registry)

| family | model (initialisation → in-domain pretraining) | input / patch | params | val BCE | val AUC | AUC SIMCT |
|---|---|---|---|---|---|---|
| **CNN, in-domain MIM** | ConvNeXtV2-B (ImageNet FCMAE → 800 ep masked-image modelling on stamps) | native 41 px, stride-1 stem | 88 M | **0.0209** | 0.99972 | 0.99914 |
| **ViT, in-domain MAE** | ViT-B (ImageNet MAE → 800 ep MAE on stamps) | 42 px / 3-px patches | 85 M | **0.0214** | 0.99972 | 0.99906 |
| Hybrid ViT + CNN | ViT-B/3 MAE + ConvNeXtV2-B MIM, fused heads | 41/42 px | 173 M | 0.0234 | 0.99972 | 0.99909 |
| ViT, other external init | DINOv2 ViT-B/14 → 800 ep stamp MAE | 42 / 3 px | 85 M | 0.0239 | 0.99966 | 0.99894 |
| ViT, scratch | ViT-B random init → 800 ep stamp MAE | 42 / 3 px | 85 M | 0.0243 | 0.99961 | 0.99873 |
| ViT, other external init | DINO v1 ViT-B → 400 ep stamp MAE | 42 / 3 px | 85 M | 0.0254 | 0.99959 | 0.99877 |
| ViT-L, astronomy init | AstroCLIP/AstroDINO ViT-L → 800 ep stamp MAE | 42 / 3 px | 303 M | 0.0263 | 0.99962 | 0.99884 |
| ViT-S | random → stamp MAE | 42 / 3 px | 21 M | 0.0263 | 0.99957 | 0.99866 |
| ViT-L | ImageNet MAE ViT-L → 800 ep stamp MAE | 42 / 3 px | 303 M | 0.0323 | 0.99931 | 0.99820 |
| CNN, MIM | ConvNeXt-T / -S, ConvNeXtV2-T, ResNet-50 → stamp MIM | native | 24–50 M | 0.026–0.034 | 0.9993–0.9996 | |
| General-purpose pretrained transformers (no stamp pretraining) | BEiT-v2, DINOv2 ViT-S/B, SigLIP, EVA-02, CLIP, DeiT-III, ViT-M SBB, XCiT | upsampled to 112–224 px | 22–93 M | 0.037–0.047 | 0.9985–0.9992 | |
| Hierarchical transformers | Swin, Swin-V2, Hiera, MaxViT, CoAtNet, GC-ViT, MViT-v2, DaViT, PVT-v2, TinyViT, CAFormer | 224 px | 20–51 M | 0.044–0.062 | 0.9976–0.9989 | |
| Galaxy-morphology models (Zoobot) | ConvNeXt nano/tiny/base, ResNet-50, EfficientNet-B0, MaxViT tiny/base (incl. Euclid-trained) | 224 px | 4–115 M | 0.039–0.072 | 0.9965–0.9991 | |
| DINOv3 | ViT-S/16, ConvNeXt-T (LVD-1689M) | 224 / native | 22–28 M | 0.046 | 0.9986 | |
| CNNs from scratch | ResNet-10/14/18/26/34/50, EfficientNetV2-S, compact lens-finder CNN | native / upsampled | 5–24 M | 0.042–0.065 | 0.9972–0.9988 | |
| Rotation-equivariant CNN | p4m group-convolution CNN | native | 6 M | 0.0444 | 0.99880 | 0.99723 |
| Astronomy foundation model | AION-1 (Polymathic AI), base, with frozen HSC tokenizer | 96 px canvas | | 0.103 | 0.9928 | |
| Set transformer | PointSetNet over the brightest point-source patches | k = 16 patches | 5 M | 0.0694 | 0.9968 | 0.9917 |
| Per-band token ViT | 5 tokens per location, one per band | 3-px patches | 21 M | 0.0764 | 0.9959 | 0.9896 |

**Key findings from the model search:**
1. **In-domain self-supervised pretraining is the decisive ingredient.** It roughly halves the BCE. With an identical
   fine-tuning recipe:
   - ViT-B with ImageNet weights only (0 stamp-pretraining epochs): 0.0433.
   - After 200 / 400 / 800 epochs of MAE on the stamps: 0.0242 / ≈0.022 / ≈0.0216.
   - ConvNeXtV2-B behaves the same way: 0 ep 0.0433 → 200 ep 0.0248 → 400 ep 0.0221.
   - In regression, pretraining length kept helping up to 1,600 epochs.
2. **External initialisation plus in-domain pretraining beats either alone.** ImageNet-MAE init → stamp MAE gave 0.0214,
   against 0.0242 for random init → stamp MAE.
3. **Resolution matters.** 3-px ViT patches or a stride-1 CNN stem keep the 2.5–10 px image separations. Standard
   16-px patches on upsampled stamps were clearly worse.
4. **Layer-wise learning-rate decay was a breakthrough for fine-tuning** (MAE ViT-B: 0.036 → 0.026 on the same
   encoder). Without it, fine-tuning overwrites the pretrained features in the lower blocks.
5. **Domain-specific foundation models did not transfer to these small stamps:** Zoobot (galaxy morphology), AstroCLIP,
   and AION-1, whose frozen tokenizer discards signal at this scale (diagnosed by measuring its reconstruction residuals).
6. **Mixing a ViT and a CNN** lowered BCE by about 9 % compared with either alone. A second model of the same family
   added only about 1 %.

### 3.2 Training techniques tested (with outcome)

**Helped:**
- Layer-wise LR decay (0.75), drop-path 0.3, weight EMA (0.999).
- Plateau restarts: roll back to the best weights and run a new warm-up/cosine cycle. Honest gain ≈ 0.0007, measured by
  selecting on one half of val and scoring on the other.
- Correlated-noise injection, which removes the noise-texture shortcut.
- Knowledge distillation from an ensemble (moderate gains, largest for CNNs).
- PSF-width and sky-level augmentation: removes about 90 % of the fragility to seeing and sky changes, at +0.0027
  clean BCE (Section 5).

**Neutral or negative:**
- Arbitrary-angle rotation augmentation.
- Flux mixup.
- Dimming augmentation.
- SAM optimiser (helped only small ResNets).
- Higher-resolution CNN feature maps.
- 2-px patches.
- MAE mask ratio 0.60 or 0.85 instead of 0.75.
- SIMCT loss up-weighting.
- Longer fine-tuning schedules.
- Refitting on train+val: the learning curve showed only about 0.0004 expected gain.

### 3.3 Ensembles

Mean of logits, scored on the 17,000 validation objects (TS = cross-fitted temperature scaling):

| ensemble | members | val BCE | val BCE after TS | AUC | TPR@1e-3 SIMCT |
|---|---|---|---|---|---|
| best single ViT | 1 | 0.0216 | 0.0211 | 0.99972 | 0.844 |
| 4 ViT encoders | 4 | 0.0206 | 0.0201 | 0.99974 | 0.872 |
| 4 ViT + 1 CNN ("best5") | 5 | 0.0201 | 0.0197 | 0.99975 | 0.870 |
| all 18 strong models | 18 | 0.0204 | 0.0198 | 0.99975 | 0.880 |

The gains saturate after about 3 diverse members.

## 4. Final frozen classifier and locked-holdout result (classification phase, frozen 2026-10-26)

**`final_kfold_plain`** = 20 models:
- 10-fold cross-validation on the 170,000 development objects × 2 families:
  - **ViT-B/3px:** ImageNet-MAE init → 400 ep stamp MAE. The 41-px stamp is bilinearly resized to 42 px.
  - **ConvNeXtV2-B:** ImageNet FCMAE init → 400 ep stamp MIM, native 41 px.
- Score = mean of the 20 models' TTA8 logits.

Locked-holdout results (30,000 objects never used for any choice, evaluated once each):

| frozen classifier | OOF / val BCE (development objects) | **holdout BCE** | holdout AUC | TPR@1e-3 SLSim | TPR@1e-3 SIMCT | flux-matched AUC |
|---|---|---|---|---|---|---|
| final_kfold_plain (20 models) | 0.0198 (OOF, 170k) | **0.0192** | 0.99978 | 0.998 | 0.906 | 0.99956 |
| final_kfold_robust (20, + PSF/sky aug) | 0.0218 (OOF, 170k) | 0.0204 | 0.99974 | 0.996 | 0.905 | 0.99949 |
| final_best5 (5 models, fixed split) | 0.0198 (val, 17k) | 0.0202 | 0.99976 | 0.997 | 0.915 | 0.99951 |
| Baseline 0 MLP (hand-crafted features) | — | — | 0.9838 (val) | — | — | — |

- Holdout ≈ validation, so there is no sign of overfitting to the validation data.
- The flux-matched AUC rules out the brightness shortcut: AUC stays ≥ 0.992 in every SIMCT flux decile.

## 5. Robustness, shortcut and transfer tests (validation data)

**Background-only test** (the inner r < 16 px replaced by noise): AUC 0.645, near chance. The models do not rely on the
background, although a model trained on the background alone reaches 0.89.

**Perturbation tests** on the 17,000 val objects: the best5 ensemble unless a row names another model; the threshold is fixed on clean data. "Clean 0.0198" is the best5 value.

| perturbation | val BCE (clean 0.0198) | SLSim false-positive rate at the clean FPR-1e-3 threshold |
|---|---|---|
| flux ×0.8 / ×1.25 | 0.026 / 0.021 | unchanged |
| noise ×1.5 | 0.032 | small rise |
| PSF +10 % / +20 % wider | 0.068 / 0.31 | 1.1 % / 5.0 % |
| sky offset +0.5σ | 0.10 | 5.0 % |
| PSF +20 %: single ViT-B (800-ep MAE) without → with PSF/sky augmentation | 0.347 → 0.041 | 5.7 % → 0.5 % |
| sky +0.5σ: the same single ViT-B without → with PSF/sky augmentation | 0.119 → 0.034 | 4.5 % → 0.2 % |

The fragility to seeing and sky level was learned rather than intrinsic: one augmentation removed about 90 % of it.

**Cross-pipeline transfer** (ViT-B/3, trained on one pipeline only):
- SLSim → SIMCT: AUC 0.985, but TPR@1e-3 falls from 0.86 to 0.59. The model learned morphology, not just brightness,
  but real-background data still needs its own training set.
- SIMCT → SLSim: AUC 0.955.

**Error analysis with the truth tables** (production out-of-fold predictions, 170k objects):
- **Hard SIMCT contaminants:** essentially all errors are **quasar pairs without a foreground galaxy** that mimic a
  lensed double: images on opposite sides, similar fluxes, faint (i ≈ 24–26.6). Their BCE is 0.23, against ≤ 0.004
  for the other contaminant types.
- **Missed lenses:** mostly SIMCT doubles (18 % missed at FPR 1e-3, against 2 % of quads), worst at small Einstein
  radius.
- **Realistic projection:** at FPR 1e-3, SIMCT purity is 0.90 at a lens fraction of 1:100 and 0.47 at
  1:1000. SIMCT recall falls from 0.898 (50 % doubles) to 0.834 if 90 % of lenses are doubles.

## 6. Production classifier: 10-fold training on all 200,000 objects (2026-11-03 → 11-16)

For the official test set, three families were each trained as 10 fold models on all 200,000 labelled objects (fold k
trains on 180k and early-stops on the other 20k):
- **A:** ViT-B/3px, 400-ep stamp MAE, 41→42 resize.
- **B:** ConvNeXtV2-B, 800-ep stamp MIM.
- **C:** ConvNeXtV2-B, 1,600-ep stamp MIM.

Score = ½ · ViT + ½ · mean(CNN B, C). It is assessed by out-of-fold (OOF) BCE: each object is scored only by fold models
that never trained on it.

| objects | n | production v1 (A + B + C) | production v2 (A2 pad42 + B + C) | pooled (A + A2 + B + C) | old frozen final_kfold_plain |
|---|---|---|---|---|---|
| all | 200,000 | 0.01846 | 0.01836 | 0.01820 | 0.01974 |
| never seen by any pretraining run | 47,000 | 0.01837 | 0.01866 | 0.01830 | 0.01920 |

Notes on the table:
- **A2** = family A with the raw 41×41 stamp zero-padded to 42 instead of resized (Section 7).
- **Old frozen final_kfold_plain** is scored with its own out-of-fold predictions on the 170k and its 20-model average
  on the former holdout 30k.
- **v1:** AUC 0.99978. On the never-seen objects: TPR@1e-3 0.998 (SLSim) and 0.907 (SIMCT).
- **Compared with the old frozen classifier on the same objects:** −6.5 % BCE on all objects, −4.3 % on never-seen
  objects.

Cost: about 12.5 GPU-hours per job (2 ViT folds per GPU) and 13–17 GPU-hours per CNN fold.

## 7. Pre-registered controlled experiments (2026-11-13 → 11-18)

**Method:** the analysis was fixed before any result:
- Primary metric: val BCE on the same 17,000 objects.
- delta = arm mean − control mean, over fine-tune seeds.
- SE = quadrature sum of the seed-to-seed SE and an object-bootstrap SE.
- Verdict: "better/worse" only beyond ±2 SE. At most one pre-declared extension of the number of seeds.

| question | models | seeds per arm | delta val BCE | delta / SE | verdict |
|---|---|---|---|---|---|
| ViT input: 41→42 bilinear resize vs zero-pad to 45 (15×15 tokens) | ViT-B/3, ImageNet-MAE → 400 ep stamp MAE, production-A recipe | 3 | +0.00028 (+1.3 %) | +0.28 | no detectable difference |
| ViT input: resize vs one-sided zero-pad to 42 (14×14 tokens) | same | 5 (extended from 3) | −0.00094 (−4.2 %) | −1.45 | no detectable difference |
| **Multi-task learning:** auxiliary heads predicting truth-table quantities (quad/double, Einstein radius, source offset, redshifts, contaminant pair type, galaxy presence, pair separation) as extra training targets | ConvNeXtV2-B, 2 stamp-MIM 800-ep encoders × 3 seeds, production-B recipe, aux weight 0.5 | 6 | **−0.00127 (−6.1 %)** | **−2.73** | **better**: all 6 multi-task runs beat all 6 controls |

- Multi-task gains are largest on SLSim (−22 % BCE) and smaller on SIMCT (−3.5 %). The auxiliary heads learn their
  targets well: quad/double accuracy 0.99–1.00, log Einstein-radius correlation 0.97–0.99.
- At production scale (200k objects, 10 folds), pad42 improved the ViT family alone by 2.5 % (−2.4 SE). The gain was
  confined to stamps seen in pretraining; on the 47k never-seen stamps there was no gain.

## 8. Lens-parameter regression (2026-10-26 → 11-15)

**Task:** for each lens, predict:
- Einstein radius θE
- source position (x, y) relative to the lens
- lens and source redshifts, each with a predicted uncertainty (Gaussian negative log-likelihood)
- the source redshift also as a 3- or 5-component Gaussian mixture
- double vs quad (BCE)

**Details:**
- Equivariant rotation/flip augmentation, with targets transformed consistently.
- Pixel-convention checks against the truth tables. A SIMCT pixel-scale error (0.168″ HSC pixels, not 0.2″) was found
  and fixed.

**Models:**
- ConvNeXtV2-B with stamp-MIM encoders of 400, 800, 1,600 and 2,400 epochs.
- ViT-B/3 with stamp-MAE encoders: dropped, because it was far worse (val NLL −0.77 against −1.00 for the CNN).
- Best: an ensemble of 7 ConvNeXtV2-B (1,600-ep encoder) runs, val NLL −1.025.

Errors of the best regression ensemble on the 4,250 val lenses per pipeline (typical = median |error| / true value;
bias = mean error / mean true value):

| parameter | SLSim typical | SLSim bias | SIMCT typical | SIMCT bias |
|---|---|---|---|---|
| Einstein radius θE | 1.1 % | −0.5 % | 2.2 % | −0.2 % |
| source offset \|β\| | 7.4 % | −1.3 % | 6.7 % | −2.5 % |
| lens redshift | 2.0 % | −0.4 % | 4.2 % | −0.5 % |
| source redshift | 1.1 % | −0.1 % | 7.6 % | −0.9 % |

- Source-position error: median 2.6 % (SLSim) and 2.3 % (SIMCT) of θE.
- Quad/double accuracy: 99.95 % (SLSim) and 99.34 % (SIMCT).

## 9. Engineering

- **Hardware:** MPCDF Viper supercomputer, AMD Instinct MI300A APUs (ROCm 6.3, PyTorch 2.9, timm 1.0), Slurm.
- **Self-supervised pretraining code:**
  - MAE for ViTs: 3-px patch re-cut of ImageNet weights to 5 bands, light decoder.
  - SimMIM/FCMAE-style masked-image modelling for CNNs.
- **Supporting code:**
  - checkpoint/resume across 24-h wall-time limits
  - automated job queues with resubmission
  - prediction stores
  - pre-registered comparison scripts
  - out-of-fold assemblers that refuse to run if any fold model's validation rows overlap its training rows
- **Compute:**
  - About 1,900 GPU-hours logged for the classification fine-tuning runs up to 2026-10-31.
  - On top of that: pretraining (ConvNeXtV2-B MIM: 36 GPU-h per 400 epochs, up to 2,400 epochs), about 50 regression
    runs, and the 40 production fold models (roughly 400–500 GPU-h).
- **Record keeping:** append-only lab notebook with hypotheses written before runs, a run registry (about 300
  classification runs), git history (about 325 commits), and logged holdout access.
- **Tools:** implemented with an AI coding agent (Claude Code) under the author's scientific direction. The author set
  the research questions, protocol, decisions and reviews.
