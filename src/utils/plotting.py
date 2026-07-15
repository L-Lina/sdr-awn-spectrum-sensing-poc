"""
Sensing plot: IQ power envelope with detected occupied regions shaded.

matplotlib is optional. If it isn't installed, plotting is skipped (not
installed automatically per project constraints) and the rest of the
pipeline still completes -- summary.csv does not depend on this succeeding.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def plot_sensing_result(iq: np.ndarray, regions: List[Tuple[int, int]], output_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed - skipping sensing plot "
              "(TODO: add matplotlib to requirements.txt in a later phase)")
        return False

    power = np.abs(iq) ** 2

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(power, linewidth=0.5, color="steelblue", label="|IQ|^2")
    for i, (start, end) in enumerate(regions):
        ax.axvspan(start, end, color="orange", alpha=0.3, label="occupied region" if i == 0 else None)
    ax.set_xlabel("sample index")
    ax.set_ylabel("power")
    ax.set_title("Spectrum sensing: energy detection")
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    print(f"[plot] saved sensing plot to {output_path}")
    return True
