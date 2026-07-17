"""
velocity_timescale.py

Tools for estimating the integral timescale from a velocity (or any
scalar) time series, computing the effective degrees of freedom (N*),
and building a confidence interval on the time-mean, following the
standard effective-degrees-of-freedom approach described in
Emery & Thomson, "Data Analysis Methods in Physical Oceanography"
(Sec. 3.15). Check the exact CI formula against your copy of the text --
the numbering/exact form isn't reproduced verbatim here.

Typical workflow
-----------------
1. check_gaps(ds)                       -> inspect time gaps / NaNs
2. rho = autocorrelation(x, nlags=...)  -> normalized, demeaned ACF
3. T_int, k_star = integral_timescale(rho, dt)
4. N_star = effective_dof(len(x), dt, T_int)
5. mean, (ci_lo, ci_hi), N_star = confidence_interval_mean(x, dt)

Notes
-----
- Autocorrelation assumes the series is demeaned, gap-free, and evenly
  sampled. Run check_gaps() first and interpolate/segment as needed.
- The integral timescale is the *area under* the ACF out to the first
  zero crossing, not the crossing lag itself:
      T_int = dt * (1 + 2 * sum_{k=1}^{k*-1} rho[k])
- u and v are treated as separate scalar series here (each gets its own
  T_int / N*), consistent with typical E&T-style mooring analysis. If
  you want a complex/rotary treatment instead, that's a different
  formulation -- ask and I can add it.
"""

import numpy as np
import xarray as xr
from scipy import stats
from scipy.signal import correlate


# ---------------------------------------------------------------------
# 1. Data QC: gaps and NaNs
# ---------------------------------------------------------------------
def check_gaps(ds, time_dim="time", variables=("u", "v"), rtol=1e-3):
    """
    Inspect an xarray Dataset for irregular time sampling and NaNs.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a time coordinate and the variables to check.
    time_dim : str
        Name of the time dimension/coordinate.
    variables : tuple of str
        Variable names to check for NaNs.
    rtol : float
        Relative tolerance (fraction of the median dt) used to flag a
        time step as irregular.

    Returns
    -------
    report : dict
        Summary of gap/NaN findings (also printed).
    """
    t = ds[time_dim].values
    dt_all = np.diff(t) / np.timedelta64(1, "s")  # seconds
    dt_median = np.median(dt_all)
    irregular = np.where(np.abs(dt_all - dt_median) > rtol * dt_median)[0]

    report = {
        "n_points": len(t),
        "dt_median_s": dt_median,
        "n_irregular_steps": len(irregular),
        "irregular_step_indices": irregular,
        "nan_counts": {},
    }

    print(f"Time series length: {report['n_points']} points")
    print(f"Median sampling interval: {dt_median:.1f} s")
    if len(irregular) > 0:
        preview = irregular[:10]
        print(f"WARNING: {len(irregular)} irregular time steps found "
              f"(indices: {preview}{'...' if len(irregular) > 10 else ''})")
    else:
        print("Time sampling is regular.")

    for var in variables:
        if var not in ds:
            continue
        n_nan = int(ds[var].isnull().sum().values)
        report["nan_counts"][var] = n_nan
        pct = 100 * n_nan / report["n_points"]
        print(f"{var}: {n_nan} NaNs ({pct:.2f}%)")

    return report


# ---------------------------------------------------------------------
# 2. Autocorrelation
# ---------------------------------------------------------------------
def autocorrelation(x, nlags=None, method="scipy"):
    """
    Normalized, mean-removed autocorrelation function (ACF) of a 1-D
    time series.

    Parameters
    ----------
    x : array_like, 1-D
        Time series values. Must not contain NaNs -- interpolate or
        segment gaps first (see check_gaps()).
    nlags : int, optional
        Maximum lag (in samples) to return. Defaults to len(x) - 1.
    method : {"scipy", "numpy"}
        Both give identical results; scipy's FFT-based correlate is
        faster for long series.

    Returns
    -------
    rho : ndarray
        Normalized ACF, rho[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    if np.any(np.isnan(x)):
        raise ValueError(
            "x contains NaNs. Interpolate or remove/segment gaps before "
            "computing the autocorrelation (see check_gaps())."
        )

    n = len(x)
    if nlags is None:
        nlags = n - 1

    x_anom = x - x.mean()  # demeaning is essential -- see module docstring

    if method == "scipy":
        full = correlate(x_anom, x_anom, mode="full", method="fft")
    else:
        full = np.correlate(x_anom, x_anom, mode="full")

    acov = full[n - 1:][: nlags + 1]  # zero lag onward
    rho = acov / acov[0]
    return rho


# ---------------------------------------------------------------------
# 3. Integral timescale
# ---------------------------------------------------------------------
def integral_timescale(rho, dt):
    """
    Estimate the integral timescale by integrating the ACF from lag
    zero to the first zero crossing:

        T_int = dt * (1 + 2 * sum_{k=1}^{k*-1} rho[k])

    Parameters
    ----------
    rho : ndarray
        Normalized ACF, rho[0] = 1.
    dt : float
        Sampling interval, in whatever time units you want T_int
        returned in (e.g. seconds, hours).

    Returns
    -------
    T_int : float
        Integral timescale, in units of dt.
    k_star : int
        Lag index of the first zero crossing (for diagnostics/plotting).
    """
    sign_changes = np.where(np.diff(np.sign(rho)) < 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(
            "ACF never crosses zero within the given lags -- increase "
            "nlags in autocorrelation()."
        )
    k_star = sign_changes[0] + 1

    T_int = dt * (1 + 2 * np.sum(rho[1:k_star]))
    return T_int, k_star


# ---------------------------------------------------------------------
# 4. Effective degrees of freedom
# ---------------------------------------------------------------------
def effective_dof(N, dt, T_int):
    """
    N* = N * dt / T_int, capped at N (can't have more independent
    samples than actual data points).
    """
    N_star = (N * dt) / T_int
    return min(N_star, N)


# ---------------------------------------------------------------------
# 5. Confidence interval on the mean
# ---------------------------------------------------------------------
def confidence_interval_mean(x, dt, T_int=None, alpha=0.05):
    """
    Confidence interval for the time-mean of x, using N* in place of N
    in the standard t-based CI. Check the exact CI form against your
    copy of Emery & Thomson (Sec. 3.15) -- this uses the standard
    t-distribution formulation, not a verbatim reproduction of a
    specific equation number.

    Parameters
    ----------
    x : array_like
        Time series (NaN-free).
    dt : float
        Sampling interval (same units used for T_int).
    T_int : float, optional
        Integral timescale. Computed internally if not supplied.
    alpha : float
        Significance level (0.05 -> 95% CI).

    Returns
    -------
    xbar : float
    (ci_lo, ci_hi) : tuple of float
    N_star : float
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if T_int is None:
        rho = autocorrelation(x, nlags=N - 1)
        T_int, _ = integral_timescale(rho, dt)

    N_star = effective_dof(N, dt, T_int)
    xbar = x.mean()
    s = x.std(ddof=1)

    se = s / np.sqrt(N_star)
    tval = stats.t.ppf(1 - alpha / 2, df=N_star - 1)

    return xbar, (xbar - tval * se, xbar + tval * se), N_star


# ---------------------------------------------------------------------
# Saunders (1987) style standard error, for cross-checking
# ---------------------------------------------------------------------
def one_sided_integral_timescale(T_int):
    """
    Convert our two-sided integral timescale (T_int, from
    integral_timescale()) to the one-sided convention used by Saunders
    (1987), T_L = integral of rho(tau) from 0 to infinity (i.e. only
    positive lags).
 
    T_int, as computed here, already sums both the positive-lag ACF and
    its mirror image (the factor of 2 in "1 + 2*sum(rho)"), so:
 
        T_L = T_int / 2
 
    Parameters
    ----------
    T_int : float
        Two-sided integral timescale, e.g. from integral_timescale().
 
    Returns
    -------
    T_L : float
        One-sided integral timescale, same units as T_int.
    """
    return T_int / 2.0
 
 
def standard_error_saunders(sigma, T_L, T_record):
    """
    Standard error of the mean following Saunders (1987):
 
        epsilon^2 = 2 * T_L * sigma^2 / T_record
 
    Parameters
    ----------
    sigma : float
        Standard deviation of the time series.
    T_L : float
        One-sided integral timescale (NOT our two-sided T_int -- see
        one_sided_integral_timescale() to convert).
    T_record : float
        Total record length (N * dt), same time units as T_L.
 
    Returns
    -------
    epsilon : float
        Standard error of the mean (same units as sigma).
    """
    epsilon_sq = 2.0 * T_L * sigma ** 2 / T_record
    return np.sqrt(epsilon_sq)
 
 
def compare_confidence_intervals(x, dt, T_int=None, alpha=0.05):
    """
    Compute the mean and its uncertainty two ways -- our N*/t-distribution
    approach, and the Saunders (1987) standard-error formula -- so you
    can check they agree once the one-sided/two-sided convention is
    handled consistently.
 
    Parameters
    ----------
    x : array_like
        Time series (NaN-free).
    dt : float
        Sampling interval, in the same time units you want everything
        reported in (e.g. seconds).
    T_int : float, optional
        Two-sided integral timescale. Computed internally if not given.
    alpha : float
        Significance level (0.05 -> 95% CI / ~1.96 sigma equivalent).
 
    Returns
    -------
    dict with both estimates, printed for comparison.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    T_record = N * dt
    sigma = x.std(ddof=1)
    xbar = x.mean()
 
    if T_int is None:
        rho = autocorrelation(x, nlags=N - 1)
        T_int, _ = integral_timescale(rho, dt)
 
    # -- our approach: N* and t-distribution CI --
    N_star = effective_dof(N, dt, T_int)
    se_ours = sigma / np.sqrt(N_star)
    tval = stats.t.ppf(1 - alpha / 2, df=N_star - 1)
    ci_ours = (xbar - tval * se_ours, xbar + tval * se_ours)
 
    # -- Saunders (1987): one-sided T_L, epsilon, normal approx --
    T_L = one_sided_integral_timescale(T_int)
    se_saunders = standard_error_saunders(sigma, T_L, T_record)
    zval = stats.norm.ppf(1 - alpha / 2)
    ci_saunders = (xbar - zval * se_saunders, xbar + zval * se_saunders)
 
    print(f"T_int (two-sided) = {T_int:.1f}, T_L (one-sided) = {T_L:.1f}")
    print(f"N* = {N_star:.1f}")
    print(f"SE (ours, N*)       = {se_ours:.5f}   -> 95% CI {ci_ours}")
    print(f"SE (Saunders, T_L)  = {se_saunders:.5f}   -> 95% CI {ci_saunders}")
    print("(SE values should match closely -- t vs. z multiplier is the "
          "main remaining difference, and matters more when N* is small)")
 
    return dict(
        mean=xbar, T_int=T_int, T_L=T_L, N_star=N_star,
        se_ours=se_ours, ci_ours=ci_ours,
        se_saunders=se_saunders, ci_saunders=ci_saunders,
    )

# ---------------------------------------------------------------------
# Emery & Thomson style trapezoidal integral timescale (one-sided)
# ---------------------------------------------------------------------
def integral_timescale_trapezoidal(rho, dt):
    """
    One-sided integral timescale using trapezoidal quadrature, following
    the discretization in Emery & Thomson:
 
        T = dt * sum_{k=0}^{m-1} [rho(k) + rho(k+1)] / 2
 
    which is equivalent to:
 
        T = dt * [rho(0)/2 + rho(1) + rho(2) + ... + rho(m-1) + rho(m)/2]
 
    Truncated at the first zero crossing (m = k_star), same cutoff
    logic as integral_timescale(). This differs from
    integral_timescale() in two ways: it is one-sided (comparable to
    Saunders' T_L, i.e. roughly half our two-sided T_int), and it uses
    trapezoidal rather than rectangle-rule weighting of the endpoint.
 
    Parameters
    ----------
    rho : ndarray
        Normalized ACF, rho[0] = 1.
    dt : float
        Sampling interval, in the time units you want T returned in.
 
    Returns
    -------
    T : float
        One-sided, trapezoidal integral timescale, in units of dt.
    k_star : int
        Lag index of the first zero crossing (cutoff used).
    """
    sign_changes = np.where(np.diff(np.sign(rho)) < 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(
            "ACF never crosses zero within the given lags -- increase "
            "nlags in autocorrelation()."
        )
    k_star = sign_changes[0] + 1
 
    pairs = rho[0:k_star] + rho[1:k_star + 1]  # [rho(k) + rho(k+1)] for k=0..k_star-1
    T = dt * np.sum(pairs) / 2.0
    return T, k_star
 
 
def compare_integral_timescales(x, dt):
    """
    Compute the integral timescale three ways on the same series, so you
    can see how much the one-sided vs. two-sided convention and the
    rectangle- vs. trapezoid-rule quadrature actually matter for your
    data:
 
      1. T_int   -- our two-sided, rectangle-rule sum
      2. T_L     -- one_sided_integral_timescale(T_int), i.e. T_int/2
      3. T_ET    -- Emery & Thomson trapezoidal, one-sided, computed
                    directly from the ACF (not derived from T_int)
 
    T_L and T_ET should be close to each other (same one-sided
    convention, different quadrature); T_int should be roughly double
    both.
 
    Parameters
    ----------
    x : array_like
        Time series (NaN-free).
    dt : float
        Sampling interval.
 
    Returns
    -------
    dict with T_int, T_L, T_ET, and the two k_star cutoffs used.
    """
    x = np.asarray(x, dtype=float)
    rho = autocorrelation(x, nlags=len(x) - 1)
 
    T_int, k_star_1 = integral_timescale(rho, dt)
    T_L = one_sided_integral_timescale(T_int)
    T_ET, k_star_2 = integral_timescale_trapezoidal(rho, dt)
 
    print(f"T_int (ours, two-sided, rectangle)     = {T_int:.1f}, k_star={k_star_1}")
    print(f"T_L   (= T_int/2, one-sided)            = {T_L:.1f}")
    print(f"T_ET  (E&T, one-sided, trapezoidal)     = {T_ET:.1f}, k_star={k_star_2}")
    print("(T_L and T_ET should be close -- remaining gap is rectangle "
          "vs. trapezoid weighting of the endpoint at the zero crossing)")
 
    return dict(T_int=T_int, T_L=T_L, T_ET=T_ET,
                 k_star_rect=k_star_1, k_star_trap=k_star_2)

# ---------------------------------------------------------------------
# Convenience wrapper: run the full pipeline on one variable of a Dataset
# ---------------------------------------------------------------------
def analyze_component(vel, varname, dt_seconds, nlags=None):
    """
    Run the full pipeline (ACF -> integral timescale -> N* -> CI) on one
    variable of an xarray Dataset and print/return a summary.
 
    Parameters
    ----------
    vel : xarray.Dataset
        Dataset containing the variable to analyze.
    varname : str
        Name of the variable, e.g. "u" or "v".
    dt_seconds : float
        Sampling interval in seconds.
    nlags : int, optional
        Max lag (in samples) to search for the zero crossing. Defaults
        to the full record if not given -- for most mooring records you
        should pass something physically motivated, e.g.
        nlags = int(10 * 86400 / dt_seconds) for a 10-day search window.
 
    Returns
    -------
    dict with T_int, k_star, N_star, mean, and ci.
    """
    x = vel[varname].values
    if np.any(np.isnan(x)):
        raise ValueError(
            f"{varname} contains NaNs -- interpolate or segment before analysis."
        )
    rho = autocorrelation(x, nlags=nlags)
    T_int, k_star = integral_timescale(rho, dt_seconds)
    N_star = effective_dof(len(x), dt_seconds, T_int)
    mean, (ci_lo, ci_hi), N_star = confidence_interval_mean(
        x, dt_seconds, T_int=T_int
    )
    print(f"\n--- {varname} ---")
    print(f"Integral timescale: {T_int:.1f} s ({T_int/3600:.2f} hr), "
          f"zero crossing at lag {k_star}")
    print(f"N = {len(x)}, N* = {N_star:.1f}")
    print(f"Mean {varname} = {mean:.4f}, 95% CI = ({ci_lo:.4f}, {ci_hi:.4f})")
    return dict(T_int=T_int, k_star=k_star, N_star=N_star,
                 mean=mean, ci=(ci_lo, ci_hi))
 
 
# ---------------------------------------------------------------------
# Example usage on an xarray Dataset with variables u, v
# (only runs if this file is executed directly, not on import)
# ---------------------------------------------------------------------
if __name__ == "__main__":
 
    # vel = xr.open_dataset("your_mooring_file.nc")
    # dt_seconds = float(vel.time.diff('time').median().values / np.timedelta64(1, 's'))
    # results = {v: analyze_component(vel, v, dt_seconds) for v in ("u", "v")}
    pass