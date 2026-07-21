"""
tidal_decomposition.py

Reusable low-pass / high-pass filtering tools for isolating tidal (and other
high-frequency) signals from velocity or other time series data.

The approach: apply a weighted rolling-window filter (default: cosine window)
to act as a low-pass filter, then subtract the low-passed signal from the
original to recover the high-frequency residual (which will include the
tidal band, given a window long enough to remove sub-inertial / mean flow
variability).

Example
-------
>>> import xarray as xr
>>> from tidal_decomposition import decompose_tidal_dataset
>>>
>>> # vel is a Dataset with "u" and "v" on dimension "time", sampled every 15 min
>>> lowpass_ds, highpass_ds = decompose_tidal_dataset(
...     vel, var_names=["u", "v"],
...     window_days=1, sampling_hours=0.25,
... )
>>> tnbd_east_filtered = lowpass_ds["u"]
>>> tnbd_east_highfreq = highpass_ds["u"]
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import windows


def compute_window_size(
    window_days: float,
    sampling_hours: float,
    force_odd: bool = True,
) -> int:
    """
    Compute the number of samples spanned by a desired window length.

    Parameters
    ----------
    window_days : float
        Desired low-pass window length, in days (e.g. 1 for a 1-day filter,
        1.5 for a 36-hour filter to remove the tidal band and near-inertial
        motions).
    sampling_hours : float
        Sampling period of the data, in hours (e.g. 0.25 for 15-min data).
    force_odd : bool, default True
        If True, ensure the returned window size is odd, which is needed for
        a symmetric, centered rolling window.

    Returns
    -------
    int
        Number of samples in the window.
    """
    window_size = int(round(window_days * 24.0 / sampling_hours))
    if force_odd and window_size % 2 == 0:
        window_size += 1
    return window_size


def make_lowpass_weights(
    window_size: int,
    window_type: str = "cosine",
) -> xr.DataArray:
    """
    Build a normalized 1-D set of weights to use as low-pass filter
    coefficients (i.e. a weighted moving average).

    Parameters
    ----------
    window_size : int
        Number of samples in the window (should be odd for a centered
        rolling window).
    window_type : str, default "cosine"
        Name of the window function. "cosine" uses scipy.signal.windows.cosine.
        Any name accepted by scipy.signal.windows.get_window also works
        (e.g. "hann", "hamming", "boxcar" for a plain moving average).

    Returns
    -------
    xr.DataArray
        1-D DataArray with dimension "window", normalized to sum to 1 so it
        behaves like a weighted moving average.
    """
    if window_type == "cosine":
        w = windows.cosine(window_size)
    else:
        w = windows.get_window(window_type, window_size)

    w = w / w.sum()
    return xr.DataArray(w, dims="window")


def lowpass_filter(
    da: xr.DataArray,
    window_size: int,
    window_type: str = "cosine",
    dim: str = "time",
    min_periods: int | None = None,
) -> xr.DataArray:
    """
    Apply a low-pass filter to a DataArray along `dim` using a centered,
    weighted rolling window.

    Parameters
    ----------
    da : xr.DataArray
        Input data (e.g. velocity component) with dimension `dim`.
    window_size : int
        Number of samples in the rolling window. Use `compute_window_size`
        to derive this from a desired window length and sampling period.
    window_type : str, default "cosine"
        Passed to `make_lowpass_weights`.
    dim : str, default "time"
        Name of the dimension to filter along.
    min_periods : int, optional
        Minimum number of observations in the window required to have a
        value (passed to `.rolling`). Defaults to requiring the full window
        (xarray's default when None).

    Returns
    -------
    xr.DataArray
        Low-pass filtered version of `da`, same shape (with NaNs at the
        edges where the window doesn't fully fit, unless min_periods is set).
    """
    weights = make_lowpass_weights(window_size, window_type=window_type)

    rolled = da.rolling(
        {dim: window_size}, center=True, min_periods=min_periods
    ).construct("window")

    filtered = rolled.dot(weights)
    filtered.name = da.name
    filtered.attrs = dict(da.attrs)
    filtered.attrs["filter"] = f"{window_type} low-pass, window={window_size} samples"
    return filtered


def highpass_residual(da: xr.DataArray, da_lowpass: xr.DataArray) -> xr.DataArray:
    """
    Compute the high-frequency residual: original minus low-pass filtered.

    This residual retains the tidal band (and any higher-frequency
    variability) that was removed by the low-pass filter.

    Parameters
    ----------
    da : xr.DataArray
        Original (unfiltered) data.
    da_lowpass : xr.DataArray
        Low-pass filtered version of `da` (e.g. from `lowpass_filter`).

    Returns
    -------
    xr.DataArray
        High-frequency residual, same shape as `da`.
    """
    residual = da - da_lowpass
    residual.name = da.name
    residual.attrs = dict(da.attrs)
    residual.attrs["filter"] = "high-pass residual (tidal + higher frequency)"
    return residual


def decompose_tidal(
    da: xr.DataArray,
    window_days: float,
    sampling_hours: float,
    window_type: str = "cosine",
    dim: str = "time",
    min_periods: int | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convenience wrapper: decompose a single DataArray into a low-pass
    (subtidal / low-frequency) component and a high-pass (tidal+) residual.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dimension `dim`.
    window_days : float
        Desired low-pass window length, in days.
    sampling_hours : float
        Sampling period of the data, in hours.
    window_type : str, default "cosine"
    dim : str, default "time"
    min_periods : int, optional

    Returns
    -------
    lowpass : xr.DataArray
    highpass : xr.DataArray
    """
    window_size = compute_window_size(window_days, sampling_hours)
    lowpass = lowpass_filter(
        da, window_size, window_type=window_type, dim=dim, min_periods=min_periods
    )
    highpass = highpass_residual(da, lowpass)
    return lowpass, highpass


def decompose_tidal_dataset(
    ds: xr.Dataset,
    var_names: list[str],
    window_days: float,
    sampling_hours: float,
    window_type: str = "cosine",
    dim: str = "time",
    min_periods: int | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Apply `decompose_tidal` to multiple variables in a Dataset at once.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variables to filter (e.g. "u", "v").
    var_names : list of str
        Names of variables in `ds` to decompose.
    window_days : float
        Desired low-pass window length, in days.
    sampling_hours : float
        Sampling period of the data, in hours.
    window_type : str, default "cosine"
    dim : str, default "time"
    min_periods : int, optional

    Returns
    -------
    lowpass_ds : xr.Dataset
        Dataset of low-pass filtered variables.
    highpass_ds : xr.Dataset
        Dataset of high-frequency (tidal+) residual variables.
    """
    lowpass_vars = {}
    highpass_vars = {}

    for var in var_names:
        lp, hp = decompose_tidal(
            ds[var],
            window_days=window_days,
            sampling_hours=sampling_hours,
            window_type=window_type,
            dim=dim,
            min_periods=min_periods,
        )
        lowpass_vars[var] = lp
        highpass_vars[var] = hp

    lowpass_ds = xr.Dataset(lowpass_vars)
    highpass_ds = xr.Dataset(highpass_vars)

    window_size = compute_window_size(window_days, sampling_hours)
    for out_ds, label in ((lowpass_ds, "low-pass"), (highpass_ds, "high-pass residual")):
        out_ds.attrs["decomposition"] = (
            f"{label} via {window_type} window, "
            f"window_days={window_days}, sampling_hours={sampling_hours}, "
            f"window_size={window_size} samples"
        )

    return lowpass_ds, highpass_ds