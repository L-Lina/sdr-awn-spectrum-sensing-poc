"""
Portable resolution for the standalone RML2016.10a dataset path used by
`experiments/*.py` batch scripts (as opposed to `run_full_experiment.py`'s
formal `--dataset-path` CLI flag in `src/utils/config.py`, which already has
no hardcoded fallback and requires an explicit value).

Project-close portability fix: every `experiments/*.py` batch script
previously hardcoded `DATASET_PATH = "/home/xiaomi/adversarial-rf/data/
RML2016.10a_dict.pkl"` as a module-level constant, tying the repo to one
specific VM. `resolve_dataset_path()` keeps that exact value as the
last-resort default (so the existing VM keeps working unmodified) while
adding two overrides above it, in priority order:

    1. an explicit `cli_value` (a script's own `--dataset-path` argument,
       where the script has one)
    2. the `SDR_AWN_DATASET_PATH` environment variable
    3. `DEFAULT_DATASET_PATH` (the historical hardcoded value, kept only as
       a fallback -- not the sole way to point at the dataset)

Does not change dataset format, sample selection, modulation/SNR handling,
checkpoint choice, preprocessing, or any experiment semantics -- path
resolution only.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
DATASET_PATH_ENV_VAR = "SDR_AWN_DATASET_PATH"


def resolve_dataset_path(cli_value: str | None = None, legacy_default: str = DEFAULT_DATASET_PATH) -> str:
    """cli_value (if truthy) > $SDR_AWN_DATASET_PATH (if set) > legacy_default."""
    if cli_value:
        return cli_value
    env_value = os.environ.get(DATASET_PATH_ENV_VAR)
    if env_value:
        return env_value
    return legacy_default


def require_dataset_path_exists(path: str) -> str:
    """Fail-fast, no silent fallback to a different dataset file."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Dataset path does not exist: {path!r}. Provide a valid path via "
            f"--dataset-path (where the script accepts it), the {DATASET_PATH_ENV_VAR} "
            f"environment variable, or place the RML2016.10a_dict.pkl file at the "
            f"legacy default ({DEFAULT_DATASET_PATH!r})."
        )
    return path
