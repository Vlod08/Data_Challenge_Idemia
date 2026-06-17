# Data Challenge Idemia — Face Occlusion Regression

Predict the fraction of a face that is occluded (224×224 image → score in
`[0, 1]`), evaluated with a gender-balanced score. This repository reproduces the
solution described in our report.

## 1. Install

```bash
uv sync          # creates .venv, installs dependencies and the `occlusion` package
```

We use [uv](https://docs.astral.sh/uv/). Every command below is run with
`uv run <script>`, which executes the script inside the project environment (no
manual activation needed).

## 2. Data

Place the challenge data here (or point to it with the
`DATA_CHALLENGE_IDEMIA_DATA_DIR` environment variable):

```
data/occlusion_datasets/
├── train.csv                 # columns: filename, FaceOcclusion, gender
├── test_students.csv         # columns: filename
└── Crop_224_5fp_100K/        # the face crops
```

## 3. Reproduce our submitted result (leaderboard score 0.00097)

The pipeline is: a DINOv2 ViT-B/14 regression model (LoRA fine-tuning), a gender
classifier, and a post-processing step ("level-down") that uses the classifier
to lower the gender-balanced score. Steps 1–2 need a GPU; steps 3–4 run on a
laptop. Each command writes its outputs under `artifacts/models/<name>/`.

```bash
# 1. Train the regression model (DINOv2 ViT-B/14 + LoRA, sigmoid head).
#    Writes model_best.pth and submission.csv (raw test predictions).
uv run scripts/train.py --config configs/lora_b_plain.yaml --override \
    experiment_name=sweep_lora_b_plain_04 batch_size=128 backbone_lr=4e-4 num_workers=16

# 2. Train the gender classifier, then predict gender on the test set.
#    predict_gender writes gender_pred.csv (probability of gender 1 per test image).
uv run scripts/train_gender.py --config configs/gender_b_lora.yaml
uv run scripts/predict_gender.py --run gender_b_lora

# 3. Save each model's predictions on its validation split (same seeded split),
#    needed to tune the post-processing offset.
uv run scripts/level_down.py dump-occ    --occ-run sweep_lora_b_plain_04
uv run scripts/level_down.py dump-gender --gender-run gender_b_lora

# 4. Tune the level-down offset on validation and write the final submission.
uv run scripts/level_down.py tune \
    --occ-val     artifacts/models/sweep_lora_b_plain_04/occ_val.csv \
    --gender-val  artifacts/models/gender_b_lora/gender_val.csv \
    --submission  artifacts/models/sweep_lora_b_plain_04/submission.csv \
    --gender-test artifacts/models/gender_b_lora/gender_pred.csv \
    --alphas 0.75
```

**Step 4 writes `submission_ld_a075.csv`** in the model folder — this is the file
we submitted. As discussed in the report, the level-down step optimises the
challenge metric rather than true fairness.

## 4. Other scripts (optional, for the analyses in the report)

```bash
# Per-gender error breakdown and calibration curve of a trained model
uv run scripts/diagnose.py --experiment-name sweep_lora_b_plain_04

# Post-hoc calibration and test-time augmentation studies (see the report)
uv run scripts/calibrate.py --help
uv run scripts/tta.py --occ-run sweep_lora_b_plain_04
```

Other training configurations (frozen / full fine-tuning, ViT-L, samplers,
augmentation) are available under `configs/` and run the same way:
`uv run scripts/train.py --config configs/<name>.yaml`. A quick smoke test on a
small subset is available with `--debug`.

## Notes

- **Metric.** `Err_g = Σ wᵢ(pᵢ − GTᵢ)² / Σ wᵢ` with `wᵢ = 1/30 + GTᵢ`, and
  `Score = (Err_0 + Err_1)/2 + |Err_0 − Err_1|`. The training loss
  (`GenderBalancedWeightedMSELoss`) mirrors this, and the best checkpoint is
  selected on the validation score.
- **Reproducibility.** All randomness is seeded (`seed`, default 42): the
  stratified train/validation split (gender × occlusion bin), the sampler, and
  the dataloader workers. All hyper-parameters are fields of `TrainConfig`
  (`src/occlusion/ml/run_config.py`).
- **Outputs.** Each run saves `config.json`, `history.csv`, `result.json`,
  `model_best.pth`, `submission.csv`, and loss curves under
  `artifacts/models/<experiment_name>/`.
