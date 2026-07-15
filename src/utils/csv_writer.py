"""Minimal CSV summary writer (stdlib csv only, no pandas dependency)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def write_summary_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError("write_summary_csv called with no rows")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] wrote {len(rows)} row(s) to {path}")
