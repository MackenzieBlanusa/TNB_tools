"""Small sanity tests for the core analysis functions.

Run from the repository root with:  pytest
Each test checks a case with a known answer, so a sign or unit slip shows up immediately.
"""
import numpy as np
import pytest
import xarray as xr

from TNB_tools.Integral_Timescale import autocorrelation, effective_dof, integral_timescale
from TNB_tools.preprocessing.magnetic_declination import (
    calculate_direction,
    calculate_true_direction,
    rotate_velocity,
)
from TNB_tools.spectra.rotary_power_spectra import auto_cross_spectra, rotary_components
from TNB_tools.tides.tidal_decomposition import compute_window_size, lowpass_filter


# ---------------------------------------------------------------- directions and rotation
@pytest.mark.parametrize("u, v, expected", [(0, 1, 0), (1, 0, 90), (0, -1, 180), (-1, 0, 270)])
def test_direction_is_toward_clockwise_from_north(u, v, expected):
    assert calculate_direction(u, v) == pytest.approx(expected)


def test_true_direction_wraps_at_360():
    assert calculate_true_direction(350.0, 20.0) == pytest.approx(10.0)
    assert calculate_true_direction(10.0, -20.0) == pytest.approx(350.0)


def test_rotation_is_counterclockwise_and_preserves_speed():
    u, v = rotate_velocity(1.0, 0.0, np.pi / 2)  # east rotated 90 deg CCW -> north
    assert (u, v) == pytest.approx((0.0, 1.0), abs=1e-12)
    rng = np.random.default_rng(0)
    u0, v0 = rng.normal(size=100), rng.normal(size=100)
    u1, v1 = rotate_velocity(u0, v0, 0.7)
    np.testing.assert_allclose(np.hypot(u1, v1), np.hypot(u0, v0))


def test_declination_correction_of_direction_matches_rotation_of_components():
    """Adding declination to a direction == rotating (u, v) clockwise by the declination."""
    dec = 135.0
    u, v = 3.0, 4.0
    ur, vr = rotate_velocity(u, v, np.deg2rad(-dec))
    assert calculate_direction(ur, vr) == pytest.approx(calculate_true_direction(calculate_direction(u, v), dec))


# ---------------------------------------------------------------- integral timescale
def _ar1(phi, n, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_autocorrelation_is_normalized():
    rho = autocorrelation(_ar1(0.5, 2000), nlags=50)
    assert rho[0] == pytest.approx(1.0)
    assert np.all(np.abs(rho) <= 1 + 1e-12)


def test_integral_timescale_of_ar1_process():
    """For AR(1) with coefficient phi, T = dt (1 + phi) / (1 - phi)."""
    phi, dt = 0.8, 1.0
    rho = autocorrelation(_ar1(phi, 100_000), nlags=500)
    T, _ = integral_timescale(rho, dt)
    assert T == pytest.approx(dt * (1 + phi) / (1 - phi), rel=0.15)


def test_autocorrelation_rejects_nans():
    with pytest.raises(ValueError):
        autocorrelation([1.0, np.nan, 2.0])


def test_effective_dof_never_exceeds_n():
    assert effective_dof(N=100, dt=1.0, T_int=0.5) == 100
    assert effective_dof(N=100, dt=1.0, T_int=10.0) == pytest.approx(10.0)


# ---------------------------------------------------------------- rotary spectra
def test_rotary_spectra_separate_ccw_and_cw_motion():
    """u = cos(wt), v = sin(wt) rotates counterclockwise; the clockwise case flips v."""
    fs = 24.0  # samples per day
    t = np.arange(0, 200, 1 / fs)  # days
    f0 = 1.0  # cycles per day
    u, v = np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)

    for vv, expect_ccw in ((v, True), (-v, False)):
        freq, Suu, Svv, Cuv = auto_cross_spectra(u, vv, fs, nperseg=24 * 20)
        S_ccw, S_cw = rotary_components(Suu, Svv, Cuv)
        k = np.argmin(np.abs(freq - f0))
        if expect_ccw:
            assert S_ccw[k] > 100 * S_cw[k]
        else:
            assert S_cw[k] > 100 * S_ccw[k]


# ---------------------------------------------------------------- tidal low-pass
def test_window_size_is_odd():
    assert compute_window_size(1, 0.25) == 97  # 96 samples per day -> next odd


def test_lowpass_keeps_mean_and_removes_semidiurnal_tide():
    dt_h = 0.25
    t = np.arange(0, 30 * 24, dt_h)  # hours
    signal = 5.0 + 10.0 * np.sin(2 * np.pi * t / 12.42)  # M2
    da = xr.DataArray(signal, dims="time")
    lp = lowpass_filter(da, compute_window_size(1.5, dt_h)).dropna("time")
    assert float(lp.mean()) == pytest.approx(5.0, abs=0.1)
    assert float(np.abs(lp - 5.0).max()) < 0.5  # tide amplitude reduced by >95%
