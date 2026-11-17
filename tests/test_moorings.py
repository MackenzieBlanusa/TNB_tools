"""Tests for the raw -> L1 standardization of mooring files, using small synthetic datasets."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TNB_tools.moorings.preprocess import l1_filename, standardize


def _raw(variables, n=10):
    t = pd.date_range("2018-03-01", periods=n, freq="10min")
    ds = xr.Dataset({k: ("time", np.asarray(v, float) * np.ones(n)) for k, v in variables.items()},
                    coords={"time": t, "z": 1100, "lat": -75.3, "lon": 164.1})
    ds.attrs["description"] = "Mooring TEST 1803 AQD"
    return ds


META = dict(raw_path="raw/TEST/x.nc", mooring="test", deployment="1803", inst_type="vel",
            position="bot", nominal_depth="1100", model="AQD", serial="", uv_units="cm/s",
            w_units="cm/s", dir_reference="true", declination="", trim_start="", trim_end="",
            include="yes", notes="")


def test_l1_filename():
    assert l1_filename("ditn", "1412", "ctd", "top", 77) == "DITN_1412_ctd_top_0077m.nc"


def test_renames_to_standard_names_and_sets_metadata():
    ds = standardize(_raw({"T": 1.0, "S": 34.8, "P": 1100.0, "spe": 10.0, "dirt": 90.0}), META)
    assert {"temp", "sal", "pres", "speed", "dir"} <= set(ds.data_vars)
    assert ds["speed"].attrs["units"] == "cm s-1"
    assert ds.attrs["mooring"] == "TEST" and ds.attrs["deployment"] == "1803"
    assert int(ds["nominal_depth"]) == 1100
    assert ds.attrs["sampling_interval_s"] == 600


def test_speed_and_direction_become_u_v():
    ds = standardize(_raw({"speed": 10.0, "dir": 90.0}), META)  # 10 cm/s toward east
    assert float(ds["u"][0]) == pytest.approx(10.0)
    assert float(ds["v"][0]) == pytest.approx(0.0, abs=1e-9)


def test_velocity_units_converted_to_cm_per_s():
    meta = dict(META, uv_units="m/s", w_units="m/s")
    ds = standardize(_raw({"u": 0.1, "v": -0.2, "w": 0.01, "speed": 22.4}), meta)
    assert float(ds["u"][0]) == pytest.approx(10.0)
    assert float(ds["v"][0]) == pytest.approx(-20.0)
    assert float(ds["w"][0]) == pytest.approx(1.0)
    assert "converted" in ds.attrs["history"]


def test_magnetic_correction_keeps_direction_and_components_consistent():
    meta = dict(META, dir_reference="magnetic", declination="135")
    ds = standardize(_raw({"u": 0.0, "v": 10.0, "speed": 10.0, "dir": 0.0}), meta)  # magnetic north
    assert float(ds["dir"][0]) == pytest.approx(135.0)
    recomputed = np.rad2deg(np.arctan2(float(ds["u"][0]), float(ds["v"][0]))) % 360
    assert recomputed == pytest.approx(135.0)


def test_all_nan_variables_dropped_and_duplicate_times_removed():
    raw = _raw({"speed": 5.0, "dir": 10.0, "temp": np.nan})
    raw = xr.concat([raw, raw.isel(time=[0])], "time")
    ds = standardize(raw, META)
    assert "temp" not in ds
    assert ds.indexes["time"].is_unique and ds.indexes["time"].is_monotonic_increasing


def test_trim():
    ds = standardize(_raw({"speed": 5.0, "dir": 10.0}), dict(META, trim_start="2018-03-01 00:30"))
    assert ds.time.values[0] == np.datetime64("2018-03-01T00:30")
