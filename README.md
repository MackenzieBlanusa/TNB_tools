## TNB-tools

Useful analysis tools for physical oceanographic research in Terra Nova Bay, Antarctica.

This repository is the shared code library. Analysis for each dissertation chapter lives in its own repository and installs this package. The analysis behind the qualifying exam paper is frozen at the tag `quals-2026`.

### Install (on each computer)

```bash
conda activate <env>
cd TNB_tools
pip install -e ".[tests]"   # editable: code changes are picked up without reinstalling
pytest                       # quick check that everything works
```

To keep notebook outputs out of git, run this once per clone:

```bash
pip install nbstripout
nbstripout --install
```

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

The hand-edited processing decisions (deployment ID, position, unit fixes, declination, trims) are kept in `metadata/moorings_manifest.csv` in this repository, so they are versioned with the code. Each L1 file records the TNB_tools git commit that made it.

**L1 file names** look like `{MOORING}_{DEPLOYMENT}_{type}_{position}_{depth}m.nc`, for example `DITD_1702_ctd_bot_1225m.nc`. DEPLOYMENT is the YYMM of the deployment start, and type is `ctd`, `temp` or `vel`. Standard variable names and units are defined in `TNB_tools/moorings/conventions.py`. All velocities are in cm/s, and directions are "toward", clockwise from true north.

**Set the data location once per computer**, using either an environment variable `TNB_MOORINGS=/path/to/data/moorings` or a one-line file `~/.config/tnbmoor/data_root.txt` containing that path.

**First time:**

```bash
python -m TNB_tools.moorings.draft_manifest <data root>/raw metadata/moorings_manifest.csv
# review the `review` column, fix units / declination / trims, then:
python -m TNB_tools.moorings.build_l1
```

**New data:** put the file in `raw/<MOORING>/`, add a row to the manifest, then run `python -m TNB_tools.moorings.build_l1`.

**In a notebook:**

```python
from TNB_tools.moorings import load

cat = load.catalog()
bot = load.series("DITD", "ctd", "bot")                  # all deployments, native sampling
vel = load.mooring_array("DITD", "vel", freq="1h")       # dims (time, position)
ctd = load.cached("DITD_ctd_1h", lambda: load.mooring_array("DITD", "ctd", "1h"))
```
