# Data Challenge Idemia — Face Occlusion Regression

Predict the fraction of a face that is occluded (224×224 image → score in
[0, 1]), evaluated with a gender-balanced score (see `task_brief.pdf`).

## Install

```bash
uv sync          # creates .venv, installs deps and the `occlusion` package (editable)
```

The package lives in `src/occlusion/` and is importable from anywhere (no
`sys.path` hacks). Data is expected at:

```
data/occlusion_datasets/
├── train.csv                 # filename, FaceOcclusion, gender
├── test_students.csv         # filename
└── Crop_224_5fp_100K/        # images (database1/2/3/...)
```

Override the data location with `DATA_CHALLENGE_IDEMIA_DATA_DIR=/path/to/data`.

## Metric & loss

- `Err = Σ wᵢ(pᵢ - GTᵢ)² / Σ wᵢ`, `wᵢ = 1/30 + GTᵢ`
- `Score = (Err_F + Err_M)/2 + |Err_F - Err_M|`

Default loss `balanced` (`GenderBalancedWeightedMSELoss`) computes the weighted
MSE per gender then averages → directly targets the Score. The best checkpoint
is selected on the **validation Score**.

## Train (end-to-end: LoRA / full / frozen)

Runs are configured by a YAML file (`configs/`), with optional CLI overrides.

```bash
# LoRA on ViT-B (registers)
uv run python scripts/train.py --config configs/lora_vitb.yaml

# Full fine-tuning ViT-L
uv run python scripts/train.py --config configs/full_vitl.yaml

# Quick edits without touching the YAML
uv run python scripts/train.py --config configs/lora_vitb.yaml \
    --override epochs=10 backbone_lr=5e-5

# Fast local smoke test (subsampled data, 2 epochs, cuda or mps)
uv run python scripts/train.py --config configs/lora_vitb.yaml --debug
```

All hyperparameters are fields of `TrainConfig`
(`src/occlusion/ml/run_config.py`): `finetune_mode {frozen,lora,full}`,
`sampler {none,gender,gender_occ}` (+`occ_power`), `augment`, `loss
{balanced,wmse}`, `scheduler`, `grad_accum`, `amp`, `tta`, `device
{auto,cuda,mps,cpu}`, …

## Fast baseline (frozen embeddings + head)

```bash
uv run python scripts/extract_embeddings.py --model-type dinov2_vits14
uv run python scripts/train_head.py --experiment-name head_v1 \
    --model-type dinov2_vits14
```

## Outputs

Per run in `artifacts/models/<experiment_name>/`: `config.json`, `history.csv`,
`loss_curve.png`, `result.json`, `model_best.pth`, `submission.csv`.
A summary row per run is appended to `artifacts/models/runs_index.csv`.

## Reproducibility

All randomness is seeded by `seed` (default 42): stratified train/val split
(gender × occlusion bin), sampler, and dataloader workers.
