"""
pvd_tools.py
------------
Generalized progressive vector diagram (PVD) utilities for velocity time
series (e.g. mooring records). Works on any subset of the data you hand it,
so you can look at the whole record or "particle track" around a specific
event.

Core idea: a PVD is just the time-integral of velocity. compute_pvd() does
that integration properly (using real elapsed time between samples, not an
assumed fixed interval), and returns a path starting at the origin. Two
patterns cover the "track back to an event" use case:

  1. SUBSET FIRST: slice u, v, time to the window you care about (e.g. 10
     days before a HSSW pulse) and call compute_pvd() on just that window.
     The path starts at (0, 0) at the beginning of your window.

  2. RECENTER ON EVENT: compute the PVD for the whole record once, then use
     recenter_pvd() to shift the path so the event sits at the origin. Then
     plot the segment of the path *before* the event index to see where the
     water came from, or *after* it to see where it goes.

Both are demonstrated at the bottom of this file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------

def _get_dt_seconds(time, dt=None):
    """Elapsed seconds between consecutive samples in `time`."""
    if time is None:
        if dt is None:
            raise ValueError("Must supply either `time` or a fixed `dt` (seconds).")
        return None  # caller will build a constant-dt array
    t = pd.to_datetime(np.asarray(time))
    return np.diff(t) / np.timedelta64(1, "s")


def compute_pvd(u, v, time=None, dt=None, distance_units="km", fill_gaps="zero"):
    """
    Integrate a velocity time series into a progressive vector diagram path.

    Parameters
    ----------
    u, v : array-like, 1D, same length
        Eastward / northward velocity components, in m/s. (If your data is
        in cm/s, e.g. many mooring RCM/ADCP products, divide by 100 first --
        do this explicitly rather than folding it into a "magic number" like
        the /10000 in the original script, which silently mixes a unit
        conversion with a distance conversion and is easy to get wrong.)
    time : array-like of datetime64-like, optional
        Timestamps for each sample. Strongly recommended even for "regular"
        900 s data, since it lets the function handle any gaps correctly
        instead of assuming every step is exactly `dt`.
    dt : float, optional
        Fixed sample interval in seconds. Required if `time` is not given.
        If both are given, `time` takes precedence unless you're
        deliberately overriding.
    distance_units : {'km', 'm'}
        Units for the returned displacement.
    fill_gaps : {'zero', 'nan', 'interpolate'}
        How to handle NaNs in u/v:
          'zero'        -> treat missing velocity as zero (parcel pauses)
          'nan'         -> propagate NaNs (path breaks/ends there)
          'interpolate' -> linearly interpolate u, v across gaps first

    Returns
    -------
    x, y : np.ndarray
        Cumulative eastward / northward displacement, same length as u,
        with x[0] = y[0] = 0.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(u)
    if len(v) != n:
        raise ValueError("u and v must be the same length.")
    if n < 2:
        raise ValueError("Need at least 2 samples to build a path.")

    if fill_gaps == "interpolate":
        u = pd.Series(u).interpolate(limit_direction="both").to_numpy()
        v = pd.Series(v).interpolate(limit_direction="both").to_numpy()
    elif fill_gaps == "zero":
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)
    elif fill_gaps != "nan":
        raise ValueError("fill_gaps must be 'zero', 'nan', or 'interpolate'")

    if time is not None:
        dt_arr = _get_dt_seconds(time, dt=dt) if dt is None else np.full(n - 1, dt)
    else:
        if dt is None:
            raise ValueError("Must supply either `time` or a fixed `dt` (seconds).")
        dt_arr = np.full(n - 1, dt)

    # Trapezoidal (average-velocity) integration between samples -- more
    # accurate than a left-endpoint cumsum, and handles uneven dt cleanly.
    du = 0.5 * (u[1:] + u[:-1]) * dt_arr
    dv = 0.5 * (v[1:] + v[:-1]) * dt_arr

    x = np.concatenate(([0.0], np.cumsum(du)))
    y = np.concatenate(([0.0], np.cumsum(dv)))

    if distance_units == "km":
        x, y = x / 1000.0, y / 1000.0
    elif distance_units != "m":
        raise ValueError("distance_units must be 'km' or 'm'")

    return x, y


def recenter_pvd(x, y, index):
    """Shift a PVD path so that point `index` sits at the origin (0, 0)."""
    return x - x[index], y - y[index]


def subset_by_time(time, u, v, start=None, end=None):
    """
    Convenience slicer: pull out the u, v, time covering [start, end].

    start / end can be anything pandas.Timestamp accepts (e.g. '2019-06-10'),
    or None to leave that side open. Useful for isolating a window around an
    event before calling compute_pvd() on just that window.
    """
    t = pd.to_datetime(np.asarray(time))
    t0 = t[0] if start is None else pd.Timestamp(start)
    t1 = t[-1] if end is None else pd.Timestamp(end)
    mask = (t >= t0) & (t <= t1)
    return t[mask], np.asarray(u)[mask], np.asarray(v)[mask]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_pvd(
    x,
    y,
    time=None,
    ax=None,
    title="Progressive Vector Diagram",
    color="C0",
    label="Water parcel path",
    mark_interval_hours=24,
    mark_color="C3",
    event_index=None,
    event_label="Event",
    distance_units="km",
    equal_aspect=True,
):
    """
    Plot a PVD path, with optional periodic time markers and an event mark.

    Parameters
    ----------
    x, y : array-like
        Output of compute_pvd (or a recentered/sliced version of it).
    time : array-like of datetime64-like, optional
        Needed to place markers at real elapsed-time intervals (e.g. every
        24 h) rather than every N samples -- important once you're using
        subsets with gaps or irregular sampling.
    mark_interval_hours : float or None
        Place a marker every N hours of elapsed time (based on `time`).
        Set to None to skip markers entirely.
    event_index : int, optional
        Index into x, y to highlight (e.g. the sample nearest a storm onset
        or HSSW pulse).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, color=color, linewidth=1.5, label=label)
    ax.plot(x[0], y[0], "o", color="green", markersize=8, label="Start", zorder=4)
    ax.plot(x[-1], y[-1], "s", color="black", markersize=8, label="End", zorder=4)

    if mark_interval_hours is not None and time is not None:
        t = pd.to_datetime(np.asarray(time))
        elapsed_hr = (t - t[0]) / np.timedelta64(1, "h")
        target_hrs = np.arange(0, elapsed_hr[-1], mark_interval_hours)
        idx = np.searchsorted(elapsed_hr, target_hrs)
        idx = idx[idx < len(x)]
        ax.plot(
            x[idx], y[idx], "o", color=mark_color, markersize=4,
            label=f"Every {mark_interval_hours:g} h", zorder=3,
        )

    if event_index is not None:
        ax.plot(
            x[event_index], y[event_index], "*", color="gold",
            markeredgecolor="k", markersize=18, label=event_label, zorder=5,
        )

    unit_label = "km" if distance_units == "km" else "m"
    ax.set_title(title)
    ax.set_xlabel(f"Eastward displacement ({unit_label})")
    ax.set_ylabel(f"Northward displacement ({unit_label})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    if equal_aspect:
        ax.set_aspect("equal")
    return ax


# --------------------------------------------------------------------------
# Example usage (edit paths/variable names for your dataset and run directly,
# or just copy the relevant block into your notebook)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import xarray as xr

    # --- Load your mooring/velocity dataset -------------------------------
    # ds = xr.open_dataset("your_mooring_file.nc")
    # u, v assumed in m/s here -- convert first if yours are in cm/s:
    # u_ms, v_ms = ds["u"].values / 100.0, ds["v"].values / 100.0
    # time = ds["time"].values

    # --- Pattern 1: whole-record PVD --------------------------------------
    # x, y = compute_pvd(u_ms, v_ms, time=time, distance_units="km")
    # plot_pvd(x, y, time=time, mark_interval_hours=24,
    #          title="Full-record Progressive Vector Diagram")
    # plt.show()

    # --- Pattern 2a: subset-first (track back over a fixed window) -------
    # t_win, u_win, v_win = subset_by_time(
    #     time, u_ms, v_ms, start="2019-06-05", end="2019-06-15"
    # )
    # x_win, y_win = compute_pvd(u_win, v_win, time=t_win, distance_units="km")
    # plot_pvd(x_win, y_win, time=t_win, mark_interval_hours=24,
    #          event_index=len(x_win) - 1, event_label="Event",
    #          title="10-day backward track into event")
    # plt.show()

    # --- Pattern 2b: recenter-on-event (whole record, event at origin) ---
    # x, y = compute_pvd(u_ms, v_ms, time=time, distance_units="km")
    # event_time = pd.Timestamp("2019-06-15T00:00")
    # event_idx = int(np.argmin(np.abs(pd.to_datetime(time) - event_time)))
    # x_c, y_c = recenter_pvd(x, y, event_idx)
    # look_back = slice(max(0, event_idx - 960), event_idx + 1)  # ~10 days @ 900s
    # plot_pvd(x_c[look_back], y_c[look_back], time=time[look_back],
    #          event_index=-1, event_label="Event",
    #          title="Backward track, recentered on event")
    # plt.show()

    print("pvd_tools loaded. See the __main__ block for usage patterns.")