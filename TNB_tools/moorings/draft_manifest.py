"""Draft manifest.csv from the existing per-mooring folders (one-off migration).

Reads file names like DITN_2014_ctd_top_77.nc and each file's attributes, fills in
everything it can, guesses velocity units from the data, and writes a `review` column
saying what a human should check. It never modifies any data file.

    python -m TNB_tools.moorings.draft_manifest /path/to/data/moorings/raw  manifest_draft.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .conventions import LEGACY_TYPE, RENAME

NAME = re.compile(r"^(?P<m>[A-Za-z0-9]+)_(?P<y>\d{4})_(?P<t>[a-z]+)_(?P<p>[a-z0-9]+)(?:_(?P<d>\d+))?\.nc$")
DESC = re.compile(r"Mooring\s+(?P<m>\S+)\s+(?P<dep>\d{4})\s+(?P<model>\S+)")


def p99(a):
    a = np.abs(np.asarray(a, float))
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 99)) if a.size else np.nan


def draft(raw_root: Path, rel_to: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw_root.rglob("*.nc")):
        m = NAME.match(f.name)
        if not m:
            print("skipped (name not recognised):", f.name)
            continue
        review = []
        with xr.open_dataset(f) as ds:
            ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})
            desc = DESC.search(str(ds.attrs.get("description", "")))
            t = ds.indexes["time"]
            deployment = desc["dep"] if desc else f"{t[0]:%y%m}"
            if not desc:
                review.append("deployment taken from first timestamp")
            model = desc["model"] if desc else ""
            if not model:
                review.append("fill in model")
            z = int(ds["z"]) if "z" in ds.variables else None
            depth = int(m["d"]) if m["d"] else z
            if m["d"] and z is not None and int(m["d"]) != z:
                review.append(f"file name depth {m['d']} != z {z}")

            uv_units = w_units = ""
            if "u" in ds:
                uv_units = "m/s" if p99(ds["u"]) < 2 and p99(ds.get("speed", ds["u"] * 0 + 99)) > 2 else "cm/s"
                if uv_units == "m/s":
                    review.append("u/v look like m/s although labelled cm/s (speed is cm/s)")
            if "w" in ds:
                w_units = "m/s" if p99(ds["w"]) < 0.5 else "cm/s"
                if w_units == "cm/s" and ds["w"].attrs.get("units") == "m/s":
                    review.append("w labelled m/s but magnitudes look like cm/s")
            if "pres" in ds and depth:
                pmed = float(ds["pres"].median())
                if abs(pmed - depth) > 0.1 * depth + 30:
                    review.append(f"median pressure {pmed:.0f} dbar far from depth {depth} m")
                if "depth" in ds:
                    dmed = float(ds["depth"].median())
                    if abs(dmed - pmed) > 0.05 * pmed + 20:
                        review.append(f"'depth' variable ({dmed:.0f} m) inconsistent with pressure ({pmed:.0f} dbar)")
            nan_frac = max((float(v.isnull().mean()) for v in ds.data_vars.values() if v.dtype.kind == "f"), default=0)
            if nan_frac > 0.3:
                review.append(f"up to {nan_frac:.0%} NaN in a variable")
            if "dir" in ds:
                review.append("confirm dir is relative to true north (declination here is large)")
            n = ds.sizes["time"]
            include = "yes"
            if n < 100:
                include = "no"
                review.append(f"only {n} samples")

        rows.append(dict(
            raw_path=str(f.relative_to(rel_to)), mooring=m["m"].upper(), deployment=deployment,
            inst_type=LEGACY_TYPE.get(m["t"], m["t"]), position=m["p"], nominal_depth=depth,
            model=model, serial="", uv_units=uv_units, w_units=w_units,
            dir_reference="true" if "dir" in ds else "", declination="",
            trim_start="", trim_end="", include=include, notes="",
            start=f"{t[0]:%Y-%m-%d}", end=f"{t[-1]:%Y-%m-%d}", review="; ".join(review),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw_root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("manifest_draft.csv")
    df = draft(raw_root, rel_to=raw_root.parent)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} files, {(df.review != '').sum()} with something to review)")
