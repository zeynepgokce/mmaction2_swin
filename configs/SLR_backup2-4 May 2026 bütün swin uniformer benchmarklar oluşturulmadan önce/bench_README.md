# Benchmark: Swin & UniFormer — Resolution Study on WLASL100 & ASLCitizen100

## Goal

Compare model performance across training and test resolutions:

| Train | Checkpoint | Eval runs |
|-------|-----------|-----------|
| Train_256 (256×256 source) | best ckpt | Test_256, Test_64, Test_SR |
| Train_64_resize256 (64×64 → bilinear 256) | best ckpt | Test_64 |

Models: **Swin-Base**, **UniFormerV2-Base**
Datasets: **WLASL100**, **ASLCitizen100**
Frames: **16 (UniformSample)**

---

## How 64→256 Bilinear Resizing Works

For any experiment where the source is 64×64:

```
decode frames (64×64)
    → Resize(scale=(256, 256), keep_ratio=False)   # OpenCV bilinear (default)
    → [train: RandomResizedCrop | test: CenterCrop(224)]
    → Resize(224, 224) [train only]
    → model input (224×224)
```

**Key point:** the model always sees 224×224 tensors. Only the upstream source resolution and interpolation history differ.

For ASLCitizen100, the source videos are at native resolution (≥256).
The `train64_resize256` variant **simulates** 64×64 by first doing `Resize(64,64)` before the upsample step — the source dataset is the same.

---

## Data Paths (TRUBA)

| Dataset | TRUBA path |
|---------|-----------|
| WLASL 256×256 | `/arf/scratch/zgokce/data/WLASL100_videos_256x256/` |
| WLASL 64×64 | `/arf/scratch/zgokce/data/wlasl100_videos_64x64/` |
| WLASL SR | `/arf/scratch/zgokce/data/wlasl100_videos_64x64_SR_vimeo90k_bd/` |
| ASLCitizen | `/arf/scratch/zgokce/data/ASL_Citizen/` |
| ASLCitizen SR | `/arf/scratch/zgokce/data/ASL_Citizen_SR/` (**TODO**: set when available) |

Annotation files are expected at `{dataset_root}/train_wlasl100_mm2.txt` (WLASL) or `train_aslcitizen100_mm2.txt` (ASLCitizen) with video subdirs `train/`, `val/`, `test/`.

---

## Config Layout

```
configs/SLR/
  bench_swin/
    swin_wlasl100_train256.py
    swin_wlasl100_train64_resize256.py
    swin_aslcitizen100_train256.py
    swin_aslcitizen100_train64_resize256.py
    eval/
      swin_wlasl100_from_train256_test256.py
      swin_wlasl100_from_train256_test64.py
      swin_wlasl100_from_train256_testSR.py
      swin_wlasl100_from_train64_test64.py
      swin_aslcitizen100_from_train256_test256.py
      swin_aslcitizen100_from_train256_test64.py
      swin_aslcitizen100_from_train256_testSR.py
      swin_aslcitizen100_from_train64_test64.py
  bench_uniformer/
    (same structure)
```

---

## SLURM Scripts Layout

```
truba/script_slurm/
  bench_swin/
    run_train_swin_wlasl100_256.sh
    run_train_swin_wlasl100_64_resize256.sh
    run_train_swin_aslcitizen100_256.sh
    run_train_swin_aslcitizen100_64_resize256.sh
    run_eval_swin_wlasl100_from_train256_test256.sh
    run_eval_swin_wlasl100_from_train256_test64.sh
    run_eval_swin_wlasl100_from_train256_testSR.sh
    run_eval_swin_wlasl100_from_train64_test64.sh
    run_eval_swin_aslcitizen100_from_train256_test256.sh
    run_eval_swin_aslcitizen100_from_train256_test64.sh
    run_eval_swin_aslcitizen100_from_train256_testSR.sh
    run_eval_swin_aslcitizen100_from_train64_test64.sh
  bench_uniformer/
    (same 12 scripts)
```

---

## How to Run — Example: Swin + WLASL100

### Step 1 — Train_256

```bash
# From mmaction2_swin repo root on TRUBA:
sbatch truba/script_slurm/bench_swin/run_train_swin_wlasl100_256.sh
```

This trains Swin-Base on WLASL 256×256 videos and immediately evaluates (Test_256) using the best checkpoint.
Workdir: `/arf/scratch/zgokce/bench/swin/wlasl100/train_256/`

### Step 2 — Eval additional test types (after training completes)

```bash
sbatch truba/script_slurm/bench_swin/run_eval_swin_wlasl100_from_train256_test64.sh
sbatch truba/script_slurm/bench_swin/run_eval_swin_wlasl100_from_train256_testSR.sh
```

### Step 3 — Train_64_resize256 (independent, can run in parallel with Step 1)

```bash
sbatch truba/script_slurm/bench_swin/run_train_swin_wlasl100_64_resize256.sh
```

### Step 4 — All additional eval scripts

```bash
# Submit all remaining eval scripts after their training jobs finish
sbatch truba/script_slurm/bench_swin/run_eval_swin_wlasl100_from_train64_test64.sh
```

---

## Report Files

All results are automatically appended by `tools/bench_report.py`:

| File | Content |
|------|---------|
| `/arf/scratch/zgokce/bench/reports/report_swin_wlasl100.txt` | Detailed blocks per run |
| `/arf/scratch/zgokce/bench/reports/report_swin_aslcitizen100.txt` | Detailed blocks per run |
| `/arf/scratch/zgokce/bench/reports/report_uniformer_wlasl100.txt` | Detailed blocks per run |
| `/arf/scratch/zgokce/bench/reports/report_uniformer_aslcitizen100.txt` | Detailed blocks per run |
| `/arf/scratch/zgokce/bench/reports/report_GLOBAL_SUMMARY.tsv` | One TSV row per eval run |

Local placeholder copies of the report files are in `work_dirs/bench/reports/`.

---

## Hyperparameters (unchanged from baseline)

| Setting | Swin-Base | UniFormerV2-Base |
|---------|-----------|-----------------|
| lr | 1e-3 (backbone ×0.1) | 2e-6 |
| weight_decay | 0.05 | 0.05 |
| optimizer | AdamW | AdamW |
| scheduler | LinearLR(2.5ep) + CosineAnnealingLR | LinearLR(1ep) + CosineAnnealingLR |
| epochs | 30 | 40 |
| batch_size | 2 | 2 |
| frames | 16 (UniformSample) | 16 (UniformSample) |
| amp | Yes (AmpOptimWrapper) | No (clip_grad=20) |
| clip_grad | — | max_norm=20 |

Changes from baseline (documented):
- **None** except dataset paths and pipeline resize steps.
- `io_backend='disk'` inlined directly (instead of via `file_client_args` variable).
- `val_pipeline = test_pipeline` aliasing used for conciseness (functionally identical).

---

## ASLCitizen SR Note

ASLCitizen100 SR data path is not yet available. The configs
`*_from_train256_testSR.py` use placeholder path `/arf/scratch/zgokce/data/ASL_Citizen_SR`.
Update `_SR_ROOT` in the respective eval configs when SR data is ready.
