"""
Rotary spectral analysis for current meter velocity time series.

Decomposes horizontal velocity (u, v) into clockwise (CW) and
counterclockwise (CCW) rotary components using the auto/cross-spectral
method of Gonella (1972).

Sign convention: CCW (+) corresponds to counterclockwise rotation,
which is the sense of pure inertial oscillations in the Southern
Hemisphere -- a useful physical check once applied to real data.

References
----------
Gonella, J. (1972). A rotary-component method for analysing
    meteorological and oceanographic vector time series.
    Deep Sea Research, 19(12), 833-846.
"""

import numpy as np
from scipy.signal import welch, csd
from scipy.stats import chi2

# ------------------------ SPECTRAL METHODS ----------------------------------------------------

def auto_cross_spectra(u, v, fs, nperseg=None, noverlap=None, detrend="constant"):
    """
    Compute auto- and cross-spectra of two orthogonal velocity
    components.

    Parameters
    ----------
    u : array_like
        Eastward (or along-axis) velocity component [cm/s or m/s].
    v : array_like
        Northward (or across-axis) velocity component, same units as u.
    fs : float
        Sampling frequency. Output freq is in the same units as fs
        (e.g. fs in cycles/day -> freq in cycles/day).
    nperseg : int, optional
        Segment length for Welch's method. Defaults to scipy's
        default (256) -- for short mooring records you'll likely want
        to set this explicitly (e.g. len(u)//4).
    noverlap : int, optional
        Overlap between segments, passed to scipy's welch/csd. Default
        None uses scipy's default (50% overlap, nperseg//2) -- this
        matches the original notebook behavior and is fine for a
        smooth point-estimate spectrum with no CI attached. Pass
        noverlap=0 for non-overlapping segments, which is required if
        you plan to compute a confidence interval on this spectrum
        (see dof_welch, spectra_chi2_ci) -- overlapping segments are
        correlated, which invalidates the independence assumption
        both CI methods rely on.
    detrend : str or False, optional
        Detrending applied to each segment before spectral estimation.
        Default 'constant' (removes segment mean).

    Returns
    -------
    freq : ndarray
        Frequencies, same units as fs.
    Suu : ndarray
        Auto-spectral density of u.
    Svv : ndarray
        Auto-spectral density of v.
    Cuv : ndarray (complex)
        Cross-spectral density of u and v.
    """
    u = np.asarray(u)
    v = np.asarray(v)

    freq, Suu = welch(u, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Svv = welch(v, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Cuv = csd(u, v, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)

    return freq, Suu, Svv, Cuv


def rotary_components(Suu, Svv, Cuv):
    """
    Decompose auto/cross-spectra into clockwise and counterclockwise
    rotary spectral components (Gonella 1972).

    Parameters
    ----------
    Suu : array_like
        Auto-spectral density of u, from auto_cross_spectra().
    Svv : array_like
        Auto-spectral density of v, from auto_cross_spectra().
    Cuv : array_like (complex)
        Cross-spectral density of u and v, from auto_cross_spectra().

    Returns
    -------
    S_ccw : ndarray
        Counterclockwise rotary power spectral density.
    S_cw : ndarray
        Clockwise rotary power spectral density.
    """
    Suu = np.asarray(Suu)
    Svv = np.asarray(Svv)
    Cuv = np.asarray(Cuv)

    S_ccw = (Suu + Svv + 2 * np.imag(Cuv)) / 4
    S_cw = (Suu + Svv - 2 * np.imag(Cuv)) / 4

    return S_ccw, S_cw


def rotary_coefficient(S_ccw, S_cw):
    """
    Gonella rotary coefficient from CCW/CW rotary spectra.

    Parameters
    ----------
    S_ccw : array_like
        Counterclockwise rotary power spectral density.
    S_cw : array_like
        Clockwise rotary power spectral density.

    Returns
    -------
    Cr : ndarray
        Rotary coefficient, range [-1, 1]. +1 = purely CCW,
        -1 = purely CW, 0 = rectilinear (no net rotation).
    """
    S_ccw = np.asarray(S_ccw)
    S_cw = np.asarray(S_cw)

    return (S_ccw - S_cw) / (S_ccw + S_cw)


def rotary_coefficient_theory(sigma, f_inertial):
    """
    Theoretical Gonella rotary coefficient for a pure inertial
    oscillation, as a function of frequency.

    Parameters
    ----------
    sigma : array_like
        Frequency array, in cycles/day, e.g.
        np.arange(10e-2, 10e1, 0.01).
    f_inertial : float
        Inertial (Coriolis) frequency, in cycles/day. Compute as:
        f_inertial = np.abs(gsw.f(latitude)) * 86400 / (2 * np.pi)

    Returns
    -------
    Cr_theory : ndarray
        Theoretical rotary coefficient, same shape as sigma.
    """
    sigma = np.asarray(sigma)
    return (2 * sigma * f_inertial) / (sigma**2 + f_inertial**2)


def ellipse_orientation(Suu, Svv, Cuv):
    """
    Orientation of the variance ellipse major axis from auto/cross-
    spectra.

    Parameters
    ----------
    Suu : array_like
        Auto-spectral density of u, from auto_cross_spectra().
    Svv : array_like
        Auto-spectral density of v, from auto_cross_spectra().
    Cuv : array_like (complex)
        Cross-spectral density of u and v, from auto_cross_spectra().

    Returns
    -------
    theta_deg : ndarray
        Orientation of the variance ellipse major axis, in degrees.
    """
    Suu = np.asarray(Suu)
    Svv = np.asarray(Svv)
    Cuv = np.asarray(Cuv)

    theta = 0.5 * np.arctan2(2 * np.real(Cuv), Suu - Svv)
    return np.rad2deg(theta)


# ------------------------------- CONFIDENCE INTERVALS BASED ON CHI^2 DISTRIBUTION AND DOF --------------------------------------------

def dof_welch(n, nperseg, noverlap=0):
    """
    Estimate degrees of freedom for a Welch-method spectral estimate.

    Parameters
    ----------
    n : int
        Total number of samples in the time series.
    nperseg : int
        Segment length used in the Welch estimate.
    noverlap : int, optional
        Overlap between segments. Default 0 (non-overlapping).

    Returns
    -------
    dof : int
        Degrees of freedom, 2 * L, where L is the number of segments.
    L : int
        Number of segments.

    Notes
    -----
    Assumes segments are statistically independent, which is only
    strictly true for noverlap=0. Overlapping segments are correlated,
    so this will overstate dof (i.e. understate uncertainty) if a
    nonzero noverlap is passed in -- use with caution in that case.
    """
    step = nperseg - noverlap
    L = (n - nperseg) // step + 1
    dof = 2 * L
    return dof, L


def spectra_chi2_ci(S_hat, dof, ci=95):
    """
    Chi-squared confidence interval for a Welch-method PSD estimate.

    Parameters
    ----------
    S_hat : array_like
        Spectral estimate (e.g. Suu, Svv, S_ccw, S_cw) at each
        frequency -- the mean over segments, as returned by
        auto_cross_spectra() or rotary_components().
    dof : int
        Degrees of freedom, from dof_welch().
    ci : float, optional
        Confidence level as a percentage. Default 95.

    Returns
    -------
    lower : ndarray
        Lower confidence bound, same shape as S_hat.
    upper : ndarray
        Upper confidence bound, same shape as S_hat.

    Notes
    -----
    Assumes the underlying process is Gaussian and segments are
    independent (see dof_welch notes on overlap). This is the standard
    approach in Emery & Thomson (2001) and Percival & Walden (1993),
    and needs far fewer segments to be reliable than the empirical
    percentile method (spectra_percentile_ci), at the cost of the
    Gaussian/independence assumptions.
    """
    S_hat = np.asarray(S_hat)
    alpha = 1 - ci / 100
    lower = dof * S_hat / chi2.ppf(1 - alpha / 2, dof)
    upper = dof * S_hat / chi2.ppf(alpha / 2, dof)
    return lower, upper

def chi2_ci_multiplier(dof, ci=95):
    """
    Frequency-independent multiplicative CI factors for a chi-squared
    confidence interval. Since dof is constant across frequency for a
    fixed Welch segmentation, lower/S_hat and upper/S_hat are also
    constant -- this is why spectral CIs are conventionally shown as
    a single bracket rather than a band spanning the whole plot.

    Returns
    -------
    f_lo, f_hi : float
        lower = S_hat * f_lo, upper = S_hat * f_hi, at any frequency.
    """
    alpha = 1 - ci / 100
    f_lo = dof / chi2.ppf(1 - alpha / 2, dof)
    f_hi = dof / chi2.ppf(alpha / 2, dof)
    return f_lo, f_hi


def add_ci_bracket(ax, f_lo, f_hi, x_frac=0.08, y_frac=0.85,
                    label='95% CI', color='k'):
    """
    Draw a representative CI bracket on a log-log PSD plot, sized
    from chi2_ci_multiplier() output. Positioned by axes-fraction
    coordinates (not tied to a specific data point), matching how
    this is typically shown in spectral literature.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on (log-log scale assumed).
    f_lo, f_hi : float
        Multiplicative CI factors from chi2_ci_multiplier().
    x_frac, y_frac : float, optional
        Position of the bracket, as a fraction (0-1) of the log-scaled
        axis range. Default places it near the upper-left.
    label : str, optional
        Text label next to the bracket.
    color : str, optional
        Line/text color.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] * (xlim[1] / xlim[0]) ** x_frac
    y_center = ylim[0] * (ylim[1] / ylim[0]) ** y_frac

    yerr_lower = y_center - y_center * f_lo
    yerr_upper = y_center * f_hi - y_center

    ax.errorbar(x_pos, y_center, yerr=[[yerr_lower], [yerr_upper]],
                fmt='none', ecolor=color, elinewidth=1.5, capsize=5)
    ax.text(x_pos * 1.4, y_center, label, va='center', fontsize=10, color=color)