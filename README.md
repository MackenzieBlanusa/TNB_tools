## TNB-tools

Useful analysis tools for physical oceanographic research in Terra Nova Bay, Antarctica.

This repository is the shared code library. Analysis for each dissertation chapter lives in its own repository and installs this package. The analysis behind the qualifying exam paper is frozen at the tag `quals-2026`.

### Install (on each computer)

```bash
cd TNB_tools
mamba env create -f environment.yml     # or conda env create; installs everything incl. TNB_tools (editable)
conda activate tnb
python -m ipykernel install --user --name tnb --display-name "Python (tnb)"
nbstripout --install                    # keep notebook outputs out of git (once per clone)
pytest                                  # quick check that everything works
```

When `environment.yml` changes, run `mamba env update -f environment.yml --prune`. Install scientific packages with conda, not pip; pip tries to compile packages such as netCDF4 from source and fails.

### Modules

| Module | What it does |
| --- | --- |
| `TNB_tools.moorings` | Raw mooring files to standardized L1 files, a catalog, and loaders for multi-deployment and gridded series |
| `TNB_tools.preprocessing.magnetic_declination` | Direction from u/v, declination correction, vector rotation |
| `TNB_tools.Integral_Timescale` | Autocorrelation, integral timescale, effective degrees of freedom, confidence intervals |
| `TNB_tools.spectra` | Auto/cross and rotary spectra, rotary coefficient, chi-squared confidence intervals |
| `TNB_tools.tides` | Low-pass / high-pass tidal decomposition, tidal–hydrographic relationships |
| `TNB_tools.trough_cross_section` | Trough cross-section geometry from bathymetry |
| `TNB_tools.pvd_tools` | Progressive vector diagram tools |

### Mooring data (`TNB_tools.moorings`)

There are three data levels, and they are never mixed:

```
<data root>/
├── raw/        files as received, one folder per mooring, never edited
├── L1/         one standardized file per instrument per deployment (made by build_l1)
├── derived/    gridded or cached products; safe to delete, they get rebuilt
└── catalog.csv one row per L1 file, generated from file attributes
```

The hand-edited processing decisions are versioned with the code in this repository:

* `metadata/moorings_manifest.csv` has one row per instrument: raw file, `raw_z` (which depth to take from a multi-instrument raw file), deployment, position, unit fixes, declination and trims.
* `metadata/deployments.csv` has one row per deployment, with lat/lon.

Each L1 file records the TNB_tools git commit that made it.

**L1 file names** look like `{MOORING}_{DEPLOYMENT}_{type}_{position}_{depth}m.nc`, for example `DITD_1702_ctd_bot_1225m.nc`. DEPLOYMENT is the YYMM of the deployment start. Type is `ctd` (SBE37), `temp` (SBE56), `vel` (Aquadopp) or `rcm` (RCM9 rotor current meter, with u/v computed from speed and direction). Standard variable names and units are defined in `TNB_tools/moorings/conventions.py`. All velocities are in cm/s, and directions are "toward", clockwise from true north.

**Set the data location once per computer**, using either an environment variable `TNB_MOORINGS=/path/to/data/moorings` or a one-line file `~/.config/tnbmoor/data_root.txt` containing that path.

**Adding raw data** (e.g. a new deployment):

```bash
# 1. copy the files, as downloaded, into <data root>/raw/<source>/<MOORING>/
# 2. draft rows for them and review the `review` column
python -m TNB_tools.moorings.draft_manifest <data root>/raw/<source> new_rows.csv
# 3. paste the reviewed rows into metadata/moorings_manifest.csv (and add the deployment to deployments.csv)
# 4. build L1 for the new rows and refresh the catalog
python -m TNB_tools.moorings.build_l1
```

Only rows without an L1 file are processed. After changing a convention, use `build_l1 --overwrite`.

**In a notebook:**

```python
from TNB_tools.moorings import load

cat = load.catalog()
bot = load.series("DITD", "ctd", "bot")                  # all deployments, native sampling
vel = load.mooring_array("DITD", "vel", freq="1h")       # dims (time, position)
ctd = load.cached("DITD_ctd_1h", lambda: load.mooring_array("DITD", "ctd", "1h"))
```
