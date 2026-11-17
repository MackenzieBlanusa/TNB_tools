"""One place that knows where the data lives, so notebooks run unchanged on the laptop and the desktop.

Set it once per machine, either:
  * environment variable:  export TNB_MOORINGS=/Users/mackenzie/Documents/Research/PhD/TNB_Research/data/moorings
  * or a one-line text file: ~/.config/tnbmoor/data_root.txt containing that path

The manifest (hand-edited processing decisions) is versioned with the code in
TNB_tools/metadata/moorings_manifest.csv; the catalog is generated and lives with the data.
"""
import os
from pathlib import Path

_CONFIG_FILE = Path.home() / ".config" / "tnbmoor" / "data_root.txt"


def data_root() -> Path:
    root = os.environ.get("TNB_MOORINGS")
    if not root and _CONFIG_FILE.exists():
        root = _CONFIG_FILE.read_text().strip()
    if not root:
        raise RuntimeError(
            "Data location not set. Set TNB_MOORINGS or write the path to "
            f"{_CONFIG_FILE}"
        )
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def raw_dir() -> Path:
    return data_root() / "raw"


def l1_dir() -> Path:
    return data_root() / "L1"


def derived_dir() -> Path:
    return data_root() / "derived"


REPO_ROOT = Path(__file__).resolve().parents[2]


def manifest_path() -> Path:
    return REPO_ROOT / "metadata" / "moorings_manifest.csv"


def catalog_path() -> Path:
    return data_root() / "catalog.csv"
