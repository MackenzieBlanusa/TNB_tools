"""Make L1 files for every manifest row that doesn't have one yet, then rebuild catalog.csv.

    python -m TNB_tools.moorings.build_l1              # only new rows
    python -m TNB_tools.moorings.build_l1 --overwrite  # redo everything (after changing conventions)

Run it after adding rows to manifest.csv for new data.
"""
import argparse
from pathlib import Path

import pandas as pd

from . import paths
from .catalog import build_catalog
from .preprocess import process_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(paths.manifest_path(), dtype=str, keep_default_na=False)
    dep = paths.deployments_path()
    deployments = pd.read_csv(dep, dtype=str, keep_default_na=False) if dep.exists() else None
    process_manifest(manifest, paths.data_root(), paths.l1_dir(), overwrite=args.overwrite,
                     deployments=deployments)
    cat = build_catalog(paths.l1_dir(), out=paths.catalog_path())
    print(f"catalog: {len(cat)} files -> {paths.catalog_path()}")


if __name__ == "__main__":
    main()
