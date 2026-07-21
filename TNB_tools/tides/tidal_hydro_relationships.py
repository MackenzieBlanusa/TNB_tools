"""
tidal_hydro_relationships.py

Tools for relating tidal-band signals to lower-frequency hydrographic
(or velocity) variability, following the approach of Bowen et al. (2021,
2023): stronger tidal currents are hypothesized to drive more mixing,
which erodes density signals on subtidal (e.g. spring-neap) timescales.

Rather than comparing raw high-frequency tidal velocity directly against
raw high-frequency T/S/density (which mostly captures advection of the
background gradient by the tide, not the mixing effect), this module:

1. Builds a smoothed "tidal energy envelope" from high-passed velocity
   components (works for u/v OR rotated along-/across-trough components,
   or any number of orthogonal components).
2. Builds low-pass / anomaly fields for hydrographic (or any other scalar)
   variables on the same subtidal timescale.
3. Cross-correlates the two, with lag, using an effective-sample-size
   correction (Chelton 1983) so significance isn't overstated for these
   strongly autocorrelated series.

Depends on `tidal_decomposition.py` (uses `lowpass_filter` and
`compute_window_size`).

Example
-------
>>> from tidal_decomposition import decompose_tidal, compute_window_size, lowpass_filter
>>> from tidal_hydro_relationships import (
...     tidal_energy_envelope, lowpass_and_anomaly, lagged_cross_correlation
... )
>>>
>>> # 1. Tidal energy envelope from rotated along-/across-trough velocity
>>> _, u_along_hp = decompose_tidal(vel.u_along, window_days=1, sampling_hours=0.25)
>>> _, u_across_hp = decompose_tidal(vel.u_across, window_days=1, sampling_hours=0.25)
>>> tke_envelope = tidal_energy_envelope(
...     u_along_hp, u_across_hp,
...     window_days=14.8, sampling_hours=0.25,  # spring-neap smoothing
... )
>>>
>>> # 2. Subtidal density anomaly
>>> rho_lowpass, rho_anomaly = lowpass_and_anomaly(
...     hydro.density, window_days=14.8, sampling_hours=0.25,
... )
>>>
>>> # 3. Cross-correlate, checking for a lag
>>> lags, corr, n_eff, conf95 = lagged_cross_correlation(
...     tke_envelope, rho_anomaly, max_lag=60  # +/- 60 samples
... )
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats

from TNB_tools.tides.tidal_decomposition import lowpass_filter, compute_window_size


# ---------------------------------------------------------------------------
# Tidal energy envelope
# ---------------------------------------------------------------------------

def compute_energy(*components: xr.DataArray) -> xr.DataArray:
    """
    Compute a kinetic-energy-like quantity from one or more velocity
    components: 0.5 * sum(component ** 2).

    Works for any orthogonal component set — e.g. (u, v), or rotated
    (along_trough, across_trough) — or even a single component if you only
    care about variance in one direction.

    Parameters
    ----------
    *components : xr.DataArray
        One or more velocity component DataArrays (already high-passed, if
        you want *tidal-band* energy specifically). Must share the same
        dimensions/coordinates.

    Returns
    -------
    xr.DataArray
        Energy-like quantity, same shape as the input components.
    """
    if len(components) == 0:
        raise ValueError("Provide at least one velocity component.")

    energy = sum(c ** 2 for c in components) * 0.5
    energy.name = "tidal_energy"
    energy.attrs["description"] = (
        f"0.5 * sum of squares of {len(components)} component(s): "
        f"{[c.name for c in components]}"
    )
    return energy


def tidal_energy_envelope(
    *components: xr.DataArray,
    window_days: float,
    sampling_hours: float,
    window_type: str = "cosine",
    dim: str = "time",
) -> xr.DataArray:
    """
    Compute a smoothed tidal energy envelope from one or more high-passed
    velocity components.

    This is `compute_energy` followed by a low-pass filter, so you get a
    slowly varying "how energetic were the tides recently" time series
    (e.g. tracking the spring-neap cycle) rather than an instantaneous,
    oscillating energy signal.

    Parameters
    ----------
    *components : xr.DataArray
        High-passed velocity components (e.g. u_highpass, v_highpass, or
        along-trough/across-trough high-passed velocities).
    window_days : float
        Smoothing window length, in days. For a spring-neap envelope, try
        something on the order of 14.8 days (the M2/S2 beat period) or a
        shorter window (e.g. 1-2 days) if you want sub-spring-neap
        variability preserved.
    sampling_hours : float
        Sampling period of the data, in hours.
    window_type : str, default "cosine"
    dim : str, default "time"

    Returns
    -------
    xr.DataArray
        Smoothed tidal energy envelope.
    """
    energy = compute_energy(*components)
    window_size = compute_window_size(window_days, sampling_hours)
    envelope = lowpass_filter(energy, window_size, window_type=window_type, dim=dim)
    envelope.name = "tidal_energy_envelope"
    envelope.attrs = dict(energy.attrs)
    envelope.attrs["envelope_smoothing"] = (
        f"{window_type} low-pass, window_days={window_days}, "
        f"sampling_hours={sampling_hours}"
    )
    return envelope


# ---------------------------------------------------------------------------
# Hydrographic (or any scalar) anomaly / low-pass
# ---------------------------------------------------------------------------

def compute_anomaly(
    da: xr.DataArray,
    dim: str = "time",
    reference: str | xr.DataArray = "mean",
) -> xr.DataArray:
    """
    Compute an anomaly for any scalar variable (salinity, temperature,
    density, etc.) relative to a reference.

    Parameters
    ----------
    da : xr.DataArray
        Input variable.
    dim : str, default "time"
        Dimension to compute the mean over, if `reference="mean"`.
    reference : "mean" or xr.DataArray, default "mean"
        If "mean", subtracts the record mean along `dim`. If a DataArray
        (e.g. a low-pass filtered version of `da`), subtracts that instead
        — use this if you want the anomaly relative to the *slowly varying*
        state rather than the full-record mean.

    Returns
    -------
    xr.DataArray
        Anomaly, same shape as `da`.
    """
    if isinstance(reference, str):
        if reference != "mean":
            raise ValueError('reference must be "mean" or an xr.DataArray')
        ref = da.mean(dim=dim)
    else:
        ref = reference

    anomaly = da - ref
    anomaly.name = da.name
    anomaly.attrs = dict(da.attrs)
    anomaly.attrs["anomaly_reference"] = (
        "record mean" if isinstance(reference, str) else "provided reference (e.g. low-pass)"
    )
    return anomaly


def lowpass_and_anomaly(
    da: xr.DataArray,
    window_days: float,
    sampling_hours: float,
    window_type: str = "cosine",
    dim: str = "time",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Low-pass filter any scalar variable (T, S, density, ...) and compute the
    anomaly relative to that low-pass (i.e. the subtidal / low-frequency
    state), rather than relative to the full-record mean.

    This is the hydrographic-variable analog of `decompose_tidal` in
    `tidal_decomposition.py` — same math, just named for this use case so
    it's clear the "high-pass" output here represents a subtidal anomaly,
    not a tidal residual.

    Parameters
    ----------
    da : xr.DataArray
        Input variable, with dimension `dim`.
    window_days : float
        Low-pass window length, in days (match this to whatever timescale
        you used for the tidal energy envelope if you're going to
        cross-correlate the two).
    sampling_hours : float
        Sampling period, in hours.
    window_type : str, default "cosine"
    dim : str, default "time"

    Returns
    -------
    lowpass : xr.DataArray
        Low-pass (subtidal) component.
    anomaly : xr.DataArray
        Anomaly relative to the low-pass component.
    """
    window_size = compute_window_size(window_days, sampling_hours)
    lowpass = lowpass_filter(da, window_size, window_type=window_type, dim=dim)
    anomaly = compute_anomaly(da, dim=dim, reference=lowpass)
    return lowpass, anomaly


# ---------------------------------------------------------------------------
# Autocorrelation, effective sample size (Chelton 1983), cross-correlation
# ---------------------------------------------------------------------------

def autocorrelation(
    da: xr.DataArray,
    dim: str = "time",
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the (biased) autocorrelation function of a 1-D DataArray for
    lags 0..max_lag.

    NaNs are dropped before computing; this assumes what remains is
    evenly sampled (true for a lowpass/envelope series with only edge NaNs
    removed — check for interior gaps first if unsure, e.g. with
    `da.isnull().sum()` and inspecting where the NaNs fall).

    Parameters
    ----------
    da : xr.DataArray
        Input series with dimension `dim`.
    dim : str, default "time"
    max_lag : int, optional
        Maximum lag (in samples) to compute. Defaults to N // 4.

    Returns
    -------
    lags : np.ndarray
        Array of lags, 0..max_lag.
    acf : np.ndarray
        Autocorrelation at each lag (acf[0] == 1.0 by definition).
    """
    x = da.dropna(dim=dim).values.astype(float)
    x = x - x.mean()
    n = len(x)

    if max_lag is None:
        max_lag = n // 4

    var = np.dot(x, x) / n
    lags = np.arange(0, max_lag + 1)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for k in lags[1:]:
        acf[k] = np.dot(x[: n - k], x[k:]) / ((n - k) * var)

    return lags, acf


def effective_sample_size(
    x: xr.DataArray,
    y: xr.DataArray,
    dim: str = "time",
    max_lag: int | None = None,
) -> float:
    """
    Estimate the effective number of independent samples (degrees of
    freedom) for a cross-correlation between two autocorrelated series,
    following Chelton (1983):

        N* = N / [ rho_x(0)*rho_y(0) + 2 * sum_{k=1}^{M} rho_x(k) * rho_y(k) ]

    where rho_x, rho_y are the autocorrelation functions of x and y.
    This accounts for the fact that autocorrelated series have far fewer
    independent samples than N, which otherwise inflates apparent
    significance of a correlation.

    Parameters
    ----------
    x, y : xr.DataArray
        The two series being correlated (should already be aligned in time
        — see `lagged_cross_correlation`, which handles this for you).
    dim : str, default "time"
    max_lag : int, optional
        Maximum lag (in samples) to include in the sum. Defaults to N // 4.
        If your data has a known decorrelation timescale, set this to a few
        multiples of it rather than relying on the default.

    Returns
    -------
    float
        Effective sample size N*.
    """
    n = min(len(x.dropna(dim=dim)), len(y.dropna(dim=dim)))

    _, acf_x = autocorrelation(x, dim=dim, max_lag=max_lag)
    _, acf_y = autocorrelation(y, dim=dim, max_lag=max_lag)

    s = acf_x[0] * acf_y[0] + 2.0 * np.sum(acf_x[1:] * acf_y[1:])
    n_eff = n / s
    return float(n_eff)


def lagged_cross_correlation(
    x: xr.DataArray,
    y: xr.DataArray,
    dim: str = "time",
    max_lag: int | None = None,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the lagged cross-correlation between two series, along with an
    effective-sample-size-corrected significance threshold (Chelton 1983).

    Positive lag means x leads y (x at time t is compared to y at time
    t + lag); negative lag means y leads x.

    Parameters
    ----------
    x, y : xr.DataArray
        The two series to cross-correlate (e.g. tidal energy envelope and
        hydrographic anomaly). Must share dimension `dim`; will be
        inner-joined on that dimension to handle any mismatched coverage.
    dim : str, default "time"
    max_lag : int, optional
        Maximum lag (in samples) in each direction. Defaults to N // 4.
    alpha : float, default 0.05
        Significance level for the confidence bound (default: 95% CI).

    Returns
    -------
    lags : np.ndarray
        Lags, from -max_lag to +max_lag (in samples — multiply by your
        sampling period to convert to time units).
    corr : np.ndarray
        Cross-correlation coefficient at each lag.
    n_eff : float
        Effective sample size used for the significance bound.
    conf_bound : float
        The +/- correlation value beyond which a correlation is significant
        at the `alpha` level, given `n_eff`.
    """
    x_clean = x.dropna(dim=dim)
    y_clean = y.dropna(dim=dim)
    x_aligned, y_aligned = xr.align(x_clean, y_clean, join="inner")

    xa = x_aligned.values.astype(float)
    ya = y_aligned.values.astype(float)
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    n = len(xa)

    if max_lag is None:
        max_lag = n // 4

    denom = np.sqrt(np.dot(xa, xa) * np.dot(ya, ya))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            xa_l = xa[: n - lag]
            ya_l = ya[lag:]
        else:
            xa_l = xa[-lag:]
            ya_l = ya[: n + lag]
        corr[i] = np.dot(xa_l, ya_l) / denom

    n_eff = effective_sample_size(x_aligned, y_aligned, dim=dim, max_lag=max_lag)
    conf_bound = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n_eff)

    return lags, corr, n_eff, conf_bound