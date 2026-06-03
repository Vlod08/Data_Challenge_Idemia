from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
from typing import Any, Mapping


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _find_project_root() -> Path:
    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").exists():
            return path
    return current


PROJECT_ROOT = Path(os.getenv("DATA_CHALLENGE_IDEMIA",
                    _find_project_root())).resolve()
_load_env_file(PROJECT_ROOT / ".env")
PROJECT_ROOT = Path(os.getenv("DATA_CHALLENGE_IDEMIA", PROJECT_ROOT)).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    train_catalog_path: Path
    test_catalog_path: Path
    crops_dir: Path
    artifacts_dir: Path
    models_dir: Path


def _build_paths() -> ProjectPaths:
    data_dir = Path(os.getenv("DATA_CHALLENGE_IDEMIA_DATA_DIR",
                    PROJECT_ROOT / "data")).resolve()
    artifacts_dir = PROJECT_ROOT / "artifacts"
    models_dir = artifacts_dir / "models"

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        train_catalog_path=Path(data_dir / "occlusion_datasets" / "train.csv"),
        test_catalog_path=Path(
            data_dir / "occlusion_datasets" / "test_students.csv"),
        crops_dir=Path(data_dir / "occlusion_datasets" / "Crop_224_5fp_100K"),
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
    )


#################################################################
#################################################################

######################### DEFAULT VALUES ########################

#################################################################
#################################################################


PATHS = _build_paths()
BATCH_SIZE = 16
EPOCHS = 50
NUM_WORKERS = 0 if os.name == "nt" else 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4  # L2 Regularization
VAL_SPLIT = 0.2
IN_CHANNELS = 3
MODELS_DIR = PATHS.models_dir
RANDOM_SEED = 42


def _to_jsonable(value: Any) -> Any:
    """Convert common training objects to values that json.dump can write."""
    if isinstance(value, argparse.Namespace):
        return _to_jsonable(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _count_model_parameters(model: Any) -> dict[str, int]:
    if not hasattr(model, "parameters"):
        return {}

    parameters = list(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }


def save_model_artifacts(
    model: Any,
    hyperparams: Mapping[str, Any] | argparse.Namespace,
    params: Mapping[str, Any] | argparse.Namespace | None = None,
    output_dir: str | Path | None = None,
    experiment_name: str | None = None,
) -> Path:
    """Save model, learned weights, hyperparameters, and run parameters.

    Returns the directory containing the saved artifacts.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to save model artifacts.") from exc

    output_dir = Path(output_dir or PATHS.models_dir)
    experiment_name = experiment_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    hyperparams_path = experiment_dir / "hyperparams.json"
    params_path = experiment_dir / "params.json"
    model_path = experiment_dir / "model.pt"
    state_dict_path = experiment_dir / "model_state_dict.pt"
    manifest_path = experiment_dir / "manifest.json"

    parameter_summary = _count_model_parameters(model)
    run_params = _to_jsonable(params or {})
    if parameter_summary:
        run_params = {
            **run_params,
            "model_parameters": parameter_summary,
        }

    with hyperparams_path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(hyperparams), file, indent=2, sort_keys=True)

    with params_path.open("w", encoding="utf-8") as file:
        json.dump(run_params, file, indent=2, sort_keys=True)

    torch.save(model, model_path)
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), state_dict_path)

    manifest = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparams_path": hyperparams_path.name,
        "params_path": params_path.name,
        "model_path": model_path.name,
        "state_dict_path": state_dict_path.name if state_dict_path.exists() else None,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    return experiment_dir


def parse_args():

    parser = argparse.ArgumentParser(description="Datachallenge train parser")
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=PATHS.models_dir)
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Unique name for this training run (used for saving models and logs)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths, file pairing, shapes, and resource settings without training")
    parser.add_argument("--weight-decay", default=WEIGHT_DECAY, type=float)
    parser.add_argument("--val-split", default=VAL_SPLIT, type=float)
    parser.add_argument("--in-channels", default=IN_CHANNELS, type=int)
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a full training checkpoint to resume from.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)

    return parser.parse_args()
