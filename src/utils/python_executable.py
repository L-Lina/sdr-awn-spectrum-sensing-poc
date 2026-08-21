"""
Portable resolution for the Python interpreter used when a script needs to
launch ANOTHER formal script as a subprocess (e.g. `experiments/
run_cfile_pipeline_smoke.py` shelling out to `run_full_experiment.py` /
`run_spectrum_sensing_utility.py` to exercise the real backend end-to-end).

Project-close portability fix, same pattern as `src/utils/dataset_path.py`:
these subprocess calls previously hardcoded `VENV_PYTHON = "/home/xiaomi/
adversarial-rf/.venv/bin/python"`, tying the repo to one specific VM's
externally-managed venv. `resolve_python_executable()` prefers the
interpreter already running the current script (`sys.executable` -- if you
launched the parent script with a Python that has torch/torchattacks
installed, its subprocesses should use the same one, no extra
configuration needed) over the historical hardcoded path, while still
allowing an explicit override for the case where the parent script itself
was NOT launched with the real-backend interpreter but the caller wants the
child subprocess(es) to use one:

    1. an explicit `cli_value` (a script's own `--python-executable`
       argument, where the script has one)
    2. the `SDR_AWN_PYTHON` environment variable
    3. `sys.executable` (the interpreter currently running this process --
       the portable default; avoids hardcoding for the common case where
       the parent script is already invoked with a torch-capable Python)
    4. `legacy_default` (the historical hardcoded venv path), only used if
       explicitly passed as `legacy_default` by a caller that wants it as a
       last-resort fallback -- never the automatic first choice.

Does not change attack/AWN/cfile pipeline logic -- interpreter selection
only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PYTHON_EXECUTABLE_ENV_VAR = "SDR_AWN_PYTHON"


def resolve_python_executable(cli_value: str | None = None, legacy_default: str | None = None) -> str:
    """cli_value (if truthy) > $SDR_AWN_PYTHON (if set) > sys.executable > legacy_default (if given)."""
    if cli_value:
        return cli_value
    env_value = os.environ.get(PYTHON_EXECUTABLE_ENV_VAR)
    if env_value:
        return env_value
    if sys.executable:
        return sys.executable
    if legacy_default:
        return legacy_default
    raise RuntimeError(
        "Could not resolve a Python interpreter: sys.executable is empty and no "
        f"--python-executable/{PYTHON_EXECUTABLE_ENV_VAR}/legacy_default was provided."
    )


def require_python_executable_exists(path: str) -> str:
    """Fail-fast, no silent fallback to a different interpreter."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Python interpreter path does not exist: {path!r}. Provide a valid path via "
            f"--python-executable (where the script accepts it) or the "
            f"{PYTHON_EXECUTABLE_ENV_VAR} environment variable."
        )
    return path
