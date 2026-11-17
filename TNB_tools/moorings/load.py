"""Loading L1 files for analysis: query the catalog, combine deployments, grid in time.

Typical use in a notebook:

    from TNB_tools.moorings import load
    cat = load.catalog()
    cat.query("mooring == 'DITD' and instrument_type == 'ctd'")

    bot = load.series("DITD", "ctd", "bot")                 # all deployments, native sampling
    ctd = load.mooring_array("DITD", "ctd", freq="1h")       # dims (time, position), hourly
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import paths
from .catalog import load_catalog

# Variables that must not be averaged directly in time.
CIRCULAR = ("dir", "heading")


def catalog() -> pd.DataFrame:
    return load_catalog(paths.catalog_path())


def select(mooring=None, inst_type=None, position=None, deployment=None, cat=None) -> pd.DataFrame:
    cat = catalog() if cat is None else cat
    m = pd.Series(True, index=cat.index)
    for col, val in (("mooring", mooring), ("instrument_type", inst_type),
                     ("position", position), ("deployment", deployment)):
        if val is not None:
            vals = [val] if isinstance(val, str) else list(val)
            m &= cat[col].astype(str).isin([str(v) for v in vals])
    return cat[m]


def open_l1(path) -> xr.Dataset:
    return xr.open_dataset(paths.l1_dir() / path)


def _tag(ds: xr.Dataset) -> xr.Dataset:
    """Turn per-file scalar metadata into time-varying coordinates so deployments can be concatenated."""
    n = ds.sizes["time"]
    return ds.drop_vars(["nominal_depth", "lat", "lon"], errors="ignore").assign_coords(
        deployment=("time", np.repeat(ds.attrs["deployment"], n)),
        nominal_depth=("time", np.repeat(float(ds.attrs["nominal_depth_m"]), n)),
    )


def series(mooring, inst_type, position, variables=None, deployments=None, cat=None) -> xr.Dataset:
    """One position on one mooring, all (or chosen) deployments joined along time at native sampling.

    Gaps between deployments are simply missing times; `deployment` and `nominal_depth`
    become coordinates along time so you can always tell which deployment a sample came from.
    """
    rows = select(mooring, inst_type, position, deployments, cat).sort_values("deployment")
    if rows.empty:
        raise KeyError(f"No L1 files for {mooring} {inst_type} {position}")
    parts = [open_l1(p) for p in rows["path"]]
    if variables is None:
        common = set.intersection(*(set(p.data_vars) for p in parts))
        dropped = set.union(*(set(p.data_vars) for p in parts)) - common
        if dropped:
            warnings.warn(f"Not in every deployment, left out: {sorted(dropped)}. Pass variables= to choose.")
        variables = sorted(common)
    parts = [_tag(p[[v for v in variables if v in p]]) for p in parts]
    out = xr.concat(parts, dim="time", join="outer", combine_attrs="drop")
    out.attrs = dict(mooring=str(mooring).upper(), instrument_type=inst_type, position=position,
                     deployments=", ".join(rows["deployment"].astype(str)),
                     source_files=", ".join(rows["path"]))
    return out


def to_grid(ds: xr.Dataset, freq: str = "1h") -> xr.Dataset:
    """Bin-average onto a regular time grid (bin labels at bin centre).

    Directions are not averaged; they are recomputed from the averaged u, v.
    Bins with no data stay NaN, so gaps between deployments remain visible.
    """
    keep = [v for v in ds.data_vars if v not in CIRCULAR]
    g = ds[keep].drop_vars(["deployment", "nominal_depth"], errors="ignore")
    g = g.resample(time=freq, label="left").mean()
    g["time"] = g["time"] + pd.Timedelta(freq) / 2
    if {"u", "v"} <= set(g.data_vars):
        g["speed"] = np.hypot(g["u"], g["v"])
        g["dir"] = (np.rad2deg(np.arctan2(g["u"], g["v"])) % 360)
    if "nominal_depth" in ds.coords:
        nd = ds["nominal_depth"].resample(time=freq, label="left").median()
        g = g.assign_coords(nominal_depth=("time", nd.values))
    g.attrs = dict(ds.attrs, time_grid=freq)
    return g


def mooring_array(mooring, inst_type, freq="1h", variables=None, positions=None, cat=None) -> xr.Dataset:
    """Every position of one instrument type on a mooring, gridded: dims (time, position).

    nominal_depth is a (time, position) coordinate because depths shift between redeployments.
    """
    sel = select(mooring, inst_type, cat=cat)
    order = ["top", "mid", "mid1", "mid2", "bot"]
    pos = positions or sorted(sel["position"].unique(), key=lambda p: order.index(p) if p in order else 99)
    grids = [to_grid(series(mooring, inst_type, p, variables=variables, cat=cat), freq) for p in pos]
    out = xr.concat(grids, dim=pd.Index(pos, name="position"), join="outer", combine_attrs="drop")
    out.attrs = dict(mooring=str(mooring).upper(), instrument_type=inst_type, time_grid=freq,
                     deployments=", ".join(sorted(sel["deployment"].astype(str).unique())))
    return out


def cached(name: str, builder, rebuild: bool = False) -> xr.Dataset:
    """Load derived/<name>.nc if it exists, otherwise build it with `builder()` and save it.

        ctd = load.cached("DITD_ctd_1h", lambda: load.mooring_array("DITD", "ctd", "1h"))
    """
    f = paths.derived_dir() / f"{name}.nc"
    if f.exists() and not rebuild:
        return xr.open_dataset(f)
    ds = builder()
    f.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(f)
    return ds


def to_datatree(moorings=None, inst_types=None, freq="1h", cat=None):
    """Optional: a DataTree view  /<mooring>/<type>  built on the fly. Nothing is stored in this shape."""
    cat = catalog() if cat is None else cat
    tree = {}
    for m in moorings or sorted(cat["mooring"].unique()):
        for t in inst_types or sorted(select(m, cat=cat)["instrument_type"].unique()):
            tree[f"/{m}/{t}"] = mooring_array(m, t, freq=freq, cat=cat)
    return xr.DataTree.from_dict(tree)
