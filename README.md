# Lensed AGN search in simulated LSST images: self-supervised ViTs and CNNs

Code for a deep-learning classifier that separates strongly lensed quasars (AGN) from look-alike contaminants in 5-band
(g r i z y) 41×41-pixel LSST first-year stamps. It was written for the
[LSST Lensed AGN Static Search Challenge](https://agnchallenge.cbpf.br). The code also includes a probabilistic
regression of lens parameters (Einstein radius, source position, redshifts, double/quad).

- **Results and the full model comparison:** [`project_summary.md`](project_summary.md).
- **The final method:** [`METHOD.md`](METHOD.md).

**Main results:**
- **Locked holdout** (30,000 objects never used for any choice): AUC 0.99978, BCE 0.0192. At a 0.1 % false-positive
  rate it recovers 99.8 % of SLSim and 90.6 % of SIMCT lenses.
- **Production ensemble** (10-fold, all 200k labelled objects): out-of-fold BCE 0.0185.
- **Main ingredient:** in-domain self-supervised pretraining on the survey stamps (MAE for ViTs, masked-image
  modelling for ConvNeXtV2), started from ImageNet self-supervised weights. ViT-B uses 3-px patches; ConvNeXtV2-B uses
  a stride-1 stem.

## Layout

| path | content |
|---|---|
| `src/lagn/` | library: data loading and normalisation, models (ViT with 3-px patches and 5-band patch embedding, native-resolution CNN stems, hybrid, band-token ViT, p4m-equivariant CNN, PointSetNet, AION wrapper), metrics, k-fold assignment |
| `scripts/` | data verification and packing, Baseline 0, self-supervised pretraining (`pretrain_mae.py`, `pretrain_mim_cnn.py`), supervised training (`train.py`; `train_aux.py` for multi-task), ensembles, robustness/shortcut tests, holdout evaluation (`eval_holdout.py`), pre-registered comparisons (`compare_inputgeom.py`) |
| `configs/` | one JSON per experiment (≈ 240): `kfold/` final recipes, `zoo/` model zoo, `inputgeom/` and `multitask/` controlled tests |
| `production/` | 10-fold training on all 200k objects (`scripts/train_prod.py`, `run_prod.py`), out-of-fold scoring (`oof.py`, `oof_v2.py`), error breakdown; `PLAN.md` is the pre-registered plan |
| `regression/` | lens-parameter regression (`scripts/train_reg.py`, targets, ensembles, error tables); `PLAN.md` |
| `jobs/` | Slurm job scripts (written for AMD MI300A nodes) and queue loops |
| `tools/submit.sh` | Slurm submission helper used by the queue scripts |

## Usage (outline)

The challenge data and truth tables are **not** included; get them from the challenge organisers. Run everything from
the repository root.

```bash
export LAGN_WORKDIR=$PWD LAGN_RAW_DIR=/path/to/all_data LAGN_VENV=/path/to/venv
python scripts/verify_pack_raw.py          # checks the FITS files
python scripts/prepare_data.py             # holdout lock, train/val split, packed .npy arrays in data/
python scripts/pretrain_mae.py --cfg configs/inputgeom/mae_pad42.json --run_id ig_pad42   # stamp MAE (+ fine-tune)
python scripts/train.py --cfg configs/kfold/vit_plain.json --run_id vit_f0 --set train.fold=0 seed=0
python scripts/eval_holdout.py --runs <run ids> --name <ensemble>   # frozen models only, once each
```

- Pretraining starts from timm weights (`vit_base_patch16_224.mae`, `convnextv2_base.fcmae`).
- Configs refer to encoders by relative paths (`experiments/<run>/encoder.pt`).
- The job scripts load site-specific modules (`rocm`, `python-waterboa`); adapt them to your cluster.

## Notes

- **Protocol:** a locked holdout read only by `scripts/eval_holdout.py` (frozen models, each evaluated once), and all
  selection on validation data.
- **Inputs:** models see pixels only. Truth tables are used only as evaluation metadata, regression targets and
  auxiliary training targets.
