"""Project paths. The data location can be overridden with the
DATA_CHALLENGE_IDEMIA_DATA_DIR environment variable (useful on the GPU host)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for path in (here, *here.parents):
        if (path / "pyproject.toml").exists():
            return path
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


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
    occ = data_dir / "occlusion_datasets"
    artifacts = PROJECT_ROOT / "artifacts"
    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        train_catalog_path=occ / "train.csv",
        test_catalog_path=occ / "test_students.csv",
        crops_dir=occ / "Crop_224_5fp_100K",
        artifacts_dir=artifacts,
        models_dir=artifacts / "models",
    )


PATHS = _build_paths()
