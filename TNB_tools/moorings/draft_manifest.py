"""Draft manifest rows for raw mooring files (run once per new batch of raw data).

Handles both layouts:
  * one instrument per file (e.g. DITN_2014_ctd_top_77.nc), and
  * one file per deployment and instrument model holding several instruments along a `z`
    dimension (e.g. the DITx files 121626.nc with description "Mooring DITN 1412 SBE37SM").

It writes one row per instrument with everything it can infer, plus a `review` column listing
what a human should check. It never modifies any data file.

    python -m TNB_tools.moorings.draft_manifest <data root>/raw/DITx  manifest_draft.csv \
        [--legacy <old folder with DITN_2014_ctd_top_77.nc-style names>]

`--legacy` is used only to copy position names (top/mid/mid1/...) from the old file names.
Without it, positions are assigned by depth order. Copy the reviewed rows into
metadata/moorings_manifest.csv.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .conventions import LEGACY_TYPE, MODEL_TYPE, RENAME

LEGACY_NAME = re.compile(r"^(?P<m>[A-Za-z0-9]+)_(?P<y>\d{4})_(?P<t>[a-z]+)_(?P<p>[a-z0-9]+)(?:_(?P<d>\d+))?\.nc$")
DESC = re.compile(r"Mooring\s+(?P<m>\S+)\s+(?P<dep>\d{4})\s+(?P<model>\S+)")
RANK_POSITIONS = {1: ["bot"], 2: ["top", "bot"], 3: ["top", "mid", "bot"], 4: ["top", "mid1", "mid2", "bot"]}


def _p99(a) -> float:
    a = np.abs(np.asarray(a, float))
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 99)) if a.size else np.nan


def legacy_positions(folder: Path | None) -> dict:
    """(mooring, year, type, depth or None) -> position, from old-style file names."""
    out = {}
    if folder is None:
        return out
    for f in Path(folder).rglob("*.nc"):
        m = LEGACY_NAME.match(f.name)
        if m:
            key = (m["m"].upper(), m["y"], LEGACY_TYPE.get(m["t"], m["t"]), int(m["d"]) if m["d"] else None)
            out[key] = m["p"]
    return out


def _checks(ds: xr.Dataset, depth: int) -> list[str]:
    review = []
    if "u" in ds and "speed" in ds and _p99(ds["u"]) < 2 and _p99(ds["speed"]) > 2:
        review.append("u/v look like m/s while speed is cm/s")
    elif "u" in ds and "speed" not in ds and _p99(ds["u"]) < 2:
        review.append("u/v magnitudes < 2: check whether units are m/s")
    if "pres" in ds:
        pmed = float(ds["pres"].median())
        if abs(pmed - depth) > 0.1 * depth + 30:
            review.append(f"median pressure {pmed:.0f} dbar far from depth {depth} m")
        if "depth" in ds:
            dmed = float(ds["depth"].median())
            if abs(dmed - pmed) > 0.05 * pmed + 20:
                review.append(f"'depth' variable ({dmed:.0f} m) inconsistent with pressure ({pmed:.0f} dbar)")
    fracs = [float(v.isnull().mean()) for v in ds.data_vars.values() if v.dtype.kind == "f"]
    if fracs and max(fracs) > 0.3:
        review.append(f"up to {max(fracs):.0%} NaN in a variable")
    if ds.sizes["time"] < 100:
        review.append(f"only {ds.sizes['time']} samples")
    return review


def _units(ds: xr.Dataset) -> tuple[str, str]:
    uv = w = ""
    if "u" in ds:
        ref = _p99(ds["speed"]) if "speed" in ds else np.nan
        uv = "m/s" if _p99(ds["u"]) < 2 and (np.isnan(ref) or ref > 2) else "cm/s"
    if "w" in ds:
        w = "m/s" if _p99(ds["w"]) < 0.5 else "cm/s"
    return uv, w


def draft(raw_root: Path, rel_to: Path, legacy: Path | None = None) -> pd.DataFrame:
    pos_lookup = legacy_positions(legacy)
    rows = []
    for f in sorted(Path(raw_root).rglob("*.nc")):
        with xr.open_dataset(f) as raw:
            if "DateTime" in raw.variables:
                raw = raw.rename({"DateTime": "time"})
            desc = DESC.search(str(raw.attrs.get("description", "")))
            legacy_name = LEGACY_NAME.match(f.name)
            if desc:
                mooring, deployment, model = desc["m"].upper(), desc["dep"], desc["model"]
            elif legacy_name:
                mooring, deployment, model = legacy_name["m"].upper(), f"{raw.indexes['time'][0]:%y%m}", ""
            else:
                print("skipped (no description attribute and unrecognised name):", f.name)
                continue
            inst_type = MODEL_TYPE.get(model.upper(), "") if model else LEGACY_TYPE.get(legacy_name["t"], "")
            zs = [int(z) for z in np.atleast_1d(raw["z"].values)] if "z" in raw.variables else [None]
            multi = "z" in raw.dims and raw.sizes["z"] > 1
            year = f"20{deployment[:2]}"
            ranks = RANK_POSITIONS.get(len(zs), [f"pos{i + 1}" for i in range(len(zs))])

            for i, z in enumerate(sorted(zs, key=lambda v: -1 if v is None else v)):
                ds = raw.sel(z=z) if multi else (raw.isel(z=0) if "z" in raw.dims else raw)
                ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})
                depth = z if z is not None else (int(legacy_name["d"]) if legacy_name and legacy_name["d"] else None)
                review = []
                if not model:
                    review.append("fill in model")
                if not inst_type:
                    review.append("unknown instrument type")
                position = pos_lookup.get((mooring, year, inst_type, depth))
                if position is None:
                    same = [k for k in pos_lookup if k[:3] == (mooring, year, inst_type)]
                    if len(same) == 1 and len(zs) == 1:
                        position = pos_lookup[same[0]]
                if position is None:
                    position = ranks[i]
                    if pos_lookup:
                        review.append("position assigned by depth order")
                if depth is None:
                    review.append("fill in nominal_depth")
                    depth = ""
                review += _checks(ds, depth or 0)
                core = [v for v in ("temp", "sal", "con", "u", "v", "speed") if v in ds]
                empty = bool(core) and all(bool(ds[v].isnull().all()) for v in core)
                if empty:
                    review.append("no valid data in " + "/".join(core))
                uv, w = _units(ds)
                t = ds.indexes["time"]
                rows.append(dict(
                    raw_path=str(f.relative_to(rel_to)), raw_z="" if not multi else z,
                    mooring=mooring, deployment=deployment, inst_type=inst_type, position=position,
                    nominal_depth=depth, model=model, serial="", uv_units=uv, w_units=w,
                    dir_reference="true" if "dir" in ds or "u" in ds else "", declination="",
                    trim_start="", trim_end="", include="no" if ds.sizes["time"] < 100 or empty else "yes", notes="",
                    start=f"{t[0]:%Y-%m-%d}", end=f"{t[-1]:%Y-%m-%d}", review="; ".join(review),
                ))
    df = pd.DataFrame(rows)
    if not df.empty:
        dup = df.duplicated(["mooring", "deployment", "inst_type", "position"], keep=False)
        df.loc[dup, "review"] = (df.loc[dup, "review"] + "; duplicate position for this type").str.strip("; ")
        df = df.sort_values(["mooring", "deployment", "inst_type", "nominal_depth"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path, help="folder of raw files (searched recursively)")
    ap.add_argument("out", type=Path, nargs="?", default=Path("manifest_draft.csv"))
    ap.add_argument("--data-root", type=Path, help="raw_path is written relative to this (default: parent of 'raw')")
    ap.add_argument("--legacy", type=Path, help="old folder whose file names carry position names")
    a = ap.parse_args()
    raw_dir = a.raw_dir.absolute()
    rel = a.data_root.absolute() if a.data_root else next((p.parent for p in [raw_dir, *raw_dir.parents] if p.name == "raw"), raw_dir.parent)
    df = draft(raw_dir, rel_to=rel, legacy=a.legacy)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} instruments, {(df.review != '').sum()} with something to review)")


if __name__ == "__main__":
    main()
