"""Raw (as-received) file -> standardized L1 file.

Everything that is specific to one file lives in one row of manifest.csv, not in code:

    raw_path        path relative to the data root, e.g. raw/DITx/DITN/121626.nc
    raw_z           depth label to select when the raw file holds several instruments along a
                    `z` dimension (e.g. 77); blank if the file holds one instrument
    mooring         DITD, DITN, DITS, TNBD, P2, P3 ...
    deployment      4-digit YYMM of deployment start, e.g. 1702
    inst_type       ctd | temp | vel
    position        top | mid | mid1 | mid2 | bot
    nominal_depth   integer metres
    model           SBE37SM, SBE56, AQD, RCM9 ...
    serial          instrument serial number (blank if unknown)
    uv_units        units u/v are actually in: cm/s or m/s
    w_units         units w is actually in: cm/s or m/s
    dir_reference   true | magnetic   (direction convention in the raw file)
    declination     degrees east, only used when dir_reference = magnetic
    trim_start      optional ISO time; drop data before (deployment/settling)
    trim_end        optional ISO time; drop data after (recovery)

Deployment-wide facts (lat, lon) come from metadata/deployments.csv, joined on mooring + deployment.
    include         yes | no
    notes           free text, copied into the file's history
"""
from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..preprocessing.magnetic_declination import calculate_true_direction, rotate_velocity
from . import conventions as C
from .paths import REPO_ROOT

L1_VERSION = "TNB_tools.moorings L1 v0.1"


def _code_version() -> str:
    """Git commit of TNB_tools used to make the file (with '-dirty' if there were uncommitted changes)."""
    import subprocess
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        sha = run("rev-parse", "--short", "HEAD")
        return (sha + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")) or "unknown"
    except Exception:
        return "unknown"


def l1_filename(mooring: str, deployment, inst_type: str, position: str, nominal_depth) -> str:
    """DITD_1702_ctd_bot_1225m.nc  (mooring_deployment_type_position_depth)."""
    return f"{mooring.upper()}_{int(deployment):04d}_{inst_type}_{position}_{int(nominal_depth):04d}m.nc"


def _blank(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""


def standardize(ds: xr.Dataset, meta: dict) -> xr.Dataset:
    """Return a standardized copy of one instrument's dataset. `meta` is one manifest row."""
    ds = ds.copy()
    history = []

    # 1. one instrument out of a multi-depth raw file, then names
    if "DateTime" in ds.variables:
        ds = ds.rename({"DateTime": "time"})
    if "z" in ds.dims:
        rz = meta.get("raw_z")
        if _blank(rz):
            if ds.sizes["z"] != 1:
                raise ValueError(f"{meta.get('raw_path')}: file has z={ds['z'].values.tolist()}, set raw_z")
            ds = ds.isel(z=0)
        else:
            ds = ds.sel(z=int(float(rz)))
        history.append(f"selected z={int(ds['z'])} from multi-instrument raw file")
    ds = ds.rename({k: v for k, v in C.RENAME.items() if k in ds.variables})
    if "z" in ds.variables:
        ds = ds.drop_vars("z")  # replaced by nominal_depth from the manifest

    # 2. time: sorted, unique
    if not ds.indexes["time"].is_monotonic_increasing:
        ds = ds.sortby("time")
        history.append("sorted time")
    dup = ds.indexes["time"].duplicated()
    if dup.any():
        ds = ds.isel(time=~dup)
        history.append(f"dropped {int(dup.sum())} duplicate times")

    # 3. trim deployment / recovery
    t0, t1 = meta.get("trim_start"), meta.get("trim_end")
    if not _blank(t0) or not _blank(t1):
        ds = ds.sel(time=slice(None if _blank(t0) else t0, None if _blank(t1) else t1))
        history.append(f"trimmed to {t0 or 'start'} .. {t1 or 'end'}")

    # 4. velocity units -> cm/s
    for names, col in ((("u", "v"), "uv_units"), (("w",), "w_units")):
        units = str(meta.get(col) or "cm/s").strip()
        if units == "m/s":
            for n in names:
                if n in ds:
                    ds[n] = ds[n] * 100.0
            history.append(f"{'/'.join(names)} converted m/s -> cm/s")

    # 5. direction to true north, and u/v for speed+dir-only current meters
    if "dir" in ds and str(meta.get("dir_reference", "true")).strip() == "magnetic":
        dec = float(meta["declination"])
        ds["dir"] = calculate_true_direction(ds["dir"], dec)
        history.append(f"dir corrected to true north with declination {dec:+.1f} deg")
        if "u" in ds:  # u/v were computed from magnetic directions too: rotate them
            # a clockwise compass correction of +dec is a mathematical rotation of -dec
            ds["u"], ds["v"] = rotate_velocity(ds["u"], ds["v"], np.deg2rad(-dec))
            history.append("u, v rotated to true north")
    if "speed" in ds and "dir" in ds and "u" not in ds:
        rad = np.deg2rad(ds["dir"])
        ds["u"] = ds["speed"] * np.sin(rad)
        ds["v"] = ds["speed"] * np.cos(rad)
        history.append("u, v computed from speed and dir")

    # 6. drop variables that are entirely NaN (e.g. RCM temp channel that never worked)
    empty = [k for k, v in ds.data_vars.items() if v.dtype.kind == "f" and bool(v.isnull().all())]
    if empty:
        ds = ds.drop_vars(empty)
        history.append(f"dropped all-NaN variables {empty}")

    # 7. per-variable attributes
    for k in ds.data_vars:
        if k in C.STANDARD_VARS:
            units, long_name = C.STANDARD_VARS[k]
            ds[k].attrs = {"units": units, "long_name": long_name}
        else:
            warnings.warn(f"{meta.get('raw_path')}: '{k}' is not a standard variable; add it to conventions.py")

    # 8. coordinates and global attributes
    lat = float(ds["lat"]) if "lat" in ds.variables else (np.nan if _blank(meta.get("lat")) else float(meta["lat"]))
    lon = float(ds["lon"]) if "lon" in ds.variables else (np.nan if _blank(meta.get("lon")) else float(meta["lon"]))
    ds = ds.drop_vars([c for c in ("lat", "lon") if c in ds.variables])
    ds = ds.assign_coords(
        lat=((), lat, {"units": "degrees_north"}),
        lon=((), lon, {"units": "degrees_east"}),
        nominal_depth=((), int(meta["nominal_depth"]), {"units": "m", "positive": "down"}),
    )

    t = ds.indexes["time"]
    dt = np.median(np.diff(t.values).astype("timedelta64[s]").astype(float)) if len(t) > 1 else np.nan
    old = dict(ds.attrs)
    ds.attrs = {
        "mooring": str(meta["mooring"]).upper(),
        "deployment": f"{int(meta['deployment']):04d}",
        "instrument_type": meta["inst_type"],
        "instrument_model": str(meta.get("model") or ""),
        "serial_number": "" if _blank(meta.get("serial")) else str(meta["serial"]),
        "position": meta["position"],
        "nominal_depth_m": int(meta["nominal_depth"]),
        "latitude": lat,
        "longitude": lon,
        "time_coverage_start": str(t[0]) if len(t) else "",
        "time_coverage_end": str(t[-1]) if len(t) else "",
        "sampling_interval_s": float(dt),
        "processing_level": "L1",
        "processing_version": L1_VERSION,
        "code_version": f"TNB_tools git {_code_version()}",
        "source_file": str(meta.get("raw_path", "")),
        "source_attributes": "; ".join(f"{k}: {str(v).strip()}" for k, v in old.items()),
        "history": f"{_dt.date.today()}: " + ("; ".join(history) or "renamed only")
        + ("" if _blank(meta.get("notes")) else f". Notes: {meta['notes']}"),
    }
    return ds


def write_l1(ds: xr.Dataset, path: Path) -> None:
    enc = {k: dict(C.ENCODING_FLOAT) for k, v in ds.data_vars.items() if v.dtype.kind == "f"}
    enc["time"] = {"units": "seconds since 1970-01-01", "dtype": "float64"}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=enc)


def process_row(meta: dict, data_root: Path, l1_root: Path, overwrite: bool = False) -> Path | None:
    out = l1_root / str(meta["mooring"]).upper() / l1_filename(
        meta["mooring"], meta["deployment"], meta["inst_type"], meta["position"], meta["nominal_depth"]
    )
    if out.exists() and not overwrite:
        return out
    with xr.open_dataset(data_root / meta["raw_path"]) as raw:
        ds = standardize(raw.load(), meta)
    if ds.sizes.get("time", 0) == 0:
        warnings.warn(f"{meta['raw_path']}: no data left after trimming, skipped")
        return None
    write_l1(ds, out)
    return out


def attach_deployments(manifest: pd.DataFrame, deployments: pd.DataFrame | None) -> pd.DataFrame:
    """Add lat/lon (and any other deployment-wide columns) to each manifest row."""
    if deployments is None or deployments.empty:
        return manifest
    m = manifest.copy()
    d = deployments.copy()
    for df in (m, d):
        df["mooring"] = df["mooring"].astype(str).str.upper()
        df["deployment"] = df["deployment"].astype(str).str.zfill(4)
    extra = [c for c in d.columns if c not in ("mooring", "deployment") and c not in m.columns]
    return m.merge(d[["mooring", "deployment", *extra]], on=["mooring", "deployment"], how="left")


def process_manifest(manifest: pd.DataFrame, data_root: Path, l1_root: Path, overwrite: bool = False,
                     deployments: pd.DataFrame | None = None) -> list[Path]:
    manifest = attach_deployments(manifest, deployments)
    done = []
    for _, row in manifest.iterrows():
        if str(row.get("include", "yes")).strip().lower() != "yes":
            continue
        p = process_row(row.to_dict(), data_root, l1_root, overwrite=overwrite)
        if p is not None:
            done.append(p)
            print("L1:", p.relative_to(l1_root))
    return done
