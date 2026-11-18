"""catalog.csv: one row per L1 file, built from the files' own attributes.

Rebuild it any time (it takes seconds); never edit it by hand. Hand-entered information
belongs in manifest.csv instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

COLUMNS = [
    "mooring", "deployment", "instrument_type", "position", "nominal_depth_m",
    "instrument_model", "serial_number", "latitude", "longitude",
    "time_coverage_start", "time_coverage_end", "sampling_interval_s",
]


def build_catalog(l1_root: Path, out: Path | None = None) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(l1_root).rglob("*.nc")):
        with xr.open_dataset(f, decode_times=False) as ds:
            row = {c: ds.attrs.get(c) for c in COLUMNS}
            row["variables"] = " ".join(ds.data_vars)
        row["path"] = str(f.relative_to(l1_root))
        rows.append(row)
    cat = pd.DataFrame(rows, columns=COLUMNS + ["variables", "path"])
    for c in ("time_coverage_start", "time_coverage_end"):
        cat[c] = pd.to_datetime(cat[c], format="mixed")
    cat = cat.sort_values(["mooring", "instrument_type", "nominal_depth_m", "deployment"]).reset_index(drop=True)
    if out is not None:
        cat.to_csv(out, index=False)
    return cat


def load_catalog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"deployment": str, "serial_number": str},
                       parse_dates=["time_coverage_start", "time_coverage_end"], date_format="mixed")
