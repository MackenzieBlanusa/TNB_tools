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


def rotary_spectra(u, v, fs, nperseg=None, detrend="constant"):
    """
    Compute clockwise/counterclockwise rotary spectra, the Gonella
    rotary coefficient, and ellipse orientation from two orthogonal
    velocity components.

    Parameters
    ----------
    u : array_like
        Eastward (or along-axis) velocity component [cm/s or m/s].
    v : array_like
        Northward (or across-axis) velocity component, same units as u.
    fs : float
        Sampling frequency. Output freq is in the same units as fs
        (e.g. fs in cycles/day -> freq in cycles/day). Use cycles/day
        if comparing against rotary_coefficient_theory(), which
        expects f_inertial in cycles/day.
    nperseg : int, optional
        Segment length for Welch's method. Defaults to scipy's
        default (256) -- for short mooring records you'll likely want
        to set this explicitly (e.g. len(u)//4).
    detrend : str or False, optional
        Detrending applied to each segment before spectral estimation.
        Default 'constant' (removes segment mean). Use 'linear' if a
        trend is present within segments, or False for no detrending.

    Returns
    -------
    freq : ndarray
        Frequencies, same units as fs.
    S_ccw : ndarray
        Counterclockwise rotary power spectral density.
    S_cw : ndarray
        Clockwise rotary power spectral density.
    Cr : ndarray
        Gonella rotary coefficient, range [-1, 1]. +1 = purely CCW,
        -1 = purely CW, 0 = rectilinear (no net rotation).
    theta_deg : ndarray
        Orientation of the variance ellipse major axis, in degrees.

    Notes
    -----
    u and v should be evenly sampled and gap-free (interpolate/fill
    gaps before calling this).
    """
    u = np.asarray(u)
    v = np.asarray(v)

    freq, Suu = welch(u, fs=fs, nperseg=nperseg, detrend=detrend)
    _, Svv = welch(v, fs=fs, nperseg=nperseg, detrend=detrend)
    _, Cuv = csd(u, v, fs=fs, nperseg=nperseg, detrend=detrend)

    S_ccw = (Suu + Svv + 2 * np.imag(Cuv)) / 4
    S_cw = (Suu + Svv - 2 * np.imag(Cuv)) / 4

    Cr = (S_ccw - S_cw) / (S_ccw + S_cw)

    theta = 0.5 * np.arctan2(2 * np.real(Cuv), Suu - Svv)
    theta_deg = np.rad2deg(theta)

    return freq, S_ccw, S_cw, Cr, theta_deg


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