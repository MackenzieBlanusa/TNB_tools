"""
Split an xarray velocity dataset into monthly or seasonal segments and
run the Gonella rotary spectral decomposition (rotary_power_spectra.py)
on each segment.

Assumes ds has a 'time' coordinate and u/v velocity DataArrays on the
same time grid (e.g. from a single mooring, one depth level at a time --
loop over depth outside this if you have a full profile).
"""

import numpy as np
import pandas as pd
import xarray as xr

from TNB_tools.spectra.rotary_power_spectra import (
    auto_cross_spectra,
    rotary_components,
    rotary_coefficient,
    dof_welch,
    spectra_chi2_ci,
)

# Southern Hemisphere austral seasons -- edit years/bounds to match your
# deployment. Kept explicit rather than using ds.time.dt.season, since
# that accessor uses DJF/MAM/JJA/SON pooling logic that assumes you want
# to combine multiple years, which isn't the case for one deployment.
DEFAULT_SEASON_BOUNDS = {
    "autumn": ("2018-03-12", "2018-05-31"),
    "winter": ("2018-06-01", "2018-08-31"),
    "spring": ("2018-09-01", "2018-11-30"),
    "summer": ("2018-12-01", "2019-01-01"),
}


def _infer_fs_cycles_per_day(time):
    """Sampling frequency in cycles/day, inferred from the median time step."""
    t = pd.to_datetime(np.asarray(time))
    dt_days = np.median(np.diff(t)) / np.timedelta64(1, "D")
    return 1.0 / dt_days


def _segment_spectrum(u, v, fs, min_frac=0.5, nperseg_frac=4, ci=95):
    """
    Run auto_cross_spectra + rotary_components on one 1-D (u, v) segment.

    nperseg is set to len(u) // nperseg_frac and noverlap=0, since
    dof_welch/spectra_chi2_ci assume independent (non-overlapping)
    segments -- see the notes in rotary_power_spectra.py.

    Returns None if the segment is too short or has too many gaps to
    trust (governed by min_frac: fraction of expected samples that must
    be present and finite).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = u.size

    finite = np.isfinite(u) & np.isfinite(v)
    if finite.sum() < min_frac * n or n < 8:
        return None

    # Short gaps: linear-interpolate rather than dropping, so the FFT
    # sees a regularly sampled series. If you have long gaps, consider
    # splitting the segment further instead of interpolating over them.
    if not finite.all():
        idx = np.arange(n)
        u = np.interp(idx, idx[finite], u[finite])
        v = np.interp(idx, idx[finite], v[finite])

    nperseg = max(8, n // nperseg_frac)
    if nperseg > n:
        nperseg = n

    freq, Suu, Svv, Cuv = auto_cross_spectra(u, v, fs, nperseg=nperseg, noverlap=0)
    S_ccw, S_cw = rotary_components(Suu, Svv, Cuv)
    Cr = rotary_coefficient(S_ccw, S_cw)

    dof, L = dof_welch(n, nperseg, noverlap=0)
    ccw_lo, ccw_hi = spectra_chi2_ci(S_ccw, dof, ci=ci)
    cw_lo, cw_hi = spectra_chi2_ci(S_cw, dof, ci=ci)

    return {
        "freq": freq,
        "Suu": Suu, "Svv": Svv, "Cuv": Cuv,
        "S_ccw": S_ccw, "S_cw": S_cw, "Cr": Cr,
        "dof": dof, "n_segments": L, "n_samples": n,
        "S_ccw_ci": (ccw_lo, ccw_hi),
        "S_cw_ci": (cw_lo, cw_hi),
    }


def monthly_rotary_spectra(ds, u_var="u", v_var="v", min_frac=0.5,
                            nperseg_frac=4, ci=95):
    """
    Resample ds into contiguous calendar months (ds.resample, NOT
    groupby -- see note below) and compute a rotary spectrum for each.

    Note on resample vs groupby: for a single ~10-month deployment you
    want sequential chunks (Mar, Apr, May, ...), not calendar months
    pooled across years. ds.groupby('time.month') would pool same-named
    months together, which is wrong here even though you only have one
    year -- resample keeps them as separate, ordered segments and
    correctly gives you partial first/last months.

    Returns
    -------
    dict keyed by the month's start Timestamp -> spectrum dict (or None
    if that month didn't have enough valid data).
    """
    fs = _infer_fs_cycles_per_day(ds.time.values)
    results = {}
    for label, seg in ds.resample(time="1MS"):
        u = seg[u_var].values
        v = seg[v_var].values
        results[label] = _segment_spectrum(u, v, fs, min_frac, nperseg_frac, ci)
    return results


def seasonal_rotary_spectra(ds, u_var="u", v_var="v",
                             season_bounds=None, min_frac=0.5,
                             nperseg_frac=4, ci=95):
    """
    Slice ds into explicit season windows and compute a rotary spectrum
    for each. Pass your own season_bounds dict of
    {name: (start_date, end_date)} -- defaults assume austral seasons
    for a Mar-2023 to Jan-2024 deployment; edit DEFAULT_SEASON_BOUNDS
    or pass season_bounds explicitly for your actual dates.

    Returns
    -------
    dict keyed by season name -> spectrum dict (or None if that season
    didn't have enough valid data).
    """
    if season_bounds is None:
        season_bounds = DEFAULT_SEASON_BOUNDS

    fs = _infer_fs_cycles_per_day(ds.time.values)
    results = {}
    for name, (start, end) in season_bounds.items():
        seg = ds.sel(time=slice(start, end))
        if seg.time.size == 0:
            results[name] = None
            continue
        u = seg[u_var].values
        v = seg[v_var].values
        results[name] = _segment_spectrum(u, v, fs, min_frac, nperseg_frac, ci)
    return results


if __name__ == "__main__":
    # Quick smoke test with synthetic data: a full year at hourly
    # resolution, u/v built from an inertial-band oscillation plus noise,
    # to confirm the monthly/seasonal split + spectrum pipeline runs
    # end-to-end before you point it at real mooring data.
    rng = np.random.default_rng(0)
    time = pd.date_range("2023-03-12", "2024-01-01", freq="1h")
    n = time.size
    t_days = np.arange(n) / 24.0

    f_inertial = 1.4  # cycles/day, plausible for high-latitude Southern Ocean
    amp = 5.0
    theta = 2 * np.pi * f_inertial * t_days
    u = amp * np.cos(theta) + rng.normal(0, 1.0, n)
    v = -amp * np.sin(theta) + rng.normal(0, 1.0, n)  # CW sense for a check

    ds = xr.Dataset(
        {"u": ("time", u), "v": ("time", v)},
        coords={"time": time},
    )

    monthly = monthly_rotary_spectra(ds)
    seasonal = seasonal_rotary_spectra(ds)

    print("Monthly segments:")
    for label, res in monthly.items():
        status = "OK" if res is not None else "skipped (insufficient data)"
        n_samp = res["n_samples"] if res else "-"
        print(f"  {pd.Timestamp(label).date()}: {status}, n={n_samp}")

    print("\nSeasonal segments:")
    for name, res in seasonal.items():
        status = "OK" if res is not None else "skipped (insufficient data)"
        n_samp = res["n_samples"] if res else "-"
        print(f"  {name}: {status}, n={n_samp}")