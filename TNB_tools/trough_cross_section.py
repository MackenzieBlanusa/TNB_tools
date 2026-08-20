"""
trough_cross_section.py
=======================
Tools for estimating HSSW cross-sectional area in the Drygalski Trough
from GEBCO bathymetry data.

Functions
---------
_cardinal_label             : compass label for a direction vector
_add_transect_endticks      : perpendicular tick marks on transect ends
_add_north_arrow            : north arrow annotation
_add_scale_bar              : scale bar annotation
extract_cross_section       : extract bathymetric transect through a mooring
find_mooring_position_on_transect : locate mooring along transect (km)
compute_hssw_area           : compute HSSW cross-sectional area and plot
plot_bathy_with_transect    : plan-view bathymetry map with transect lines

Notes
-----
Bathymetry source: GEBCO 2023 (15 arcsecond resolution).
GEBCO treats floating ice (ice shelves, ice tongues) as land — elevations
are set to the ice surface height above sea level, not the sub-ice seafloor.
If your transect crosses the Drygalski Ice Tongue you will see a land
barrier where there is actually ocean beneath the ice.

To resolve sub-ice bathymetry, swap GEBCO for BedMachine Antarctica v3
(Morlighem et al. 2020, NSIDC). BedMachine provides:
    - 'bed'      : actual seafloor including sub-shelf cavities (m)
    - 'mask'     : 0=ocean, 1=land, 2=grounded ice, 3=floating ice, 4=lake
BedMachine uses a polar stereographic grid (EPSG:3031) so coordinates
must be reprojected with pyproj before interpolation. A drop-in replacement
for extract_cross_section using BedMachine can be added when needed.

Dependencies
------------
numpy, xarray, scipy, matplotlib
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _cardinal_label(dlon, dlat):
    """
    Return a compass label (N, NE, E, SE, S, SW, W, NW) for a
    direction vector (dlon, dlat).
    """
    angle = np.rad2deg(np.arctan2(dlat, dlon))   # CCW from East
    dirs  = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    idx   = int((angle + 22.5) % 360 / 45)
    return dirs[idx]


def _add_transect_endticks(ax, lons, lats, color='#D85A30', ticklen=0.015):
    """
    Perpendicular tick marks at both ends of a transect line.
    Standard cartographic convention indicating a finite cross-section face.
    """
    dlon = lons[-1] - lons[0]
    dlat = lats[-1] - lats[0]
    norm = np.sqrt(dlon**2 + dlat**2)
    tx, ty = dlon / norm, dlat / norm
    px, py = -ty, tx
    for lon_end, lat_end in [(lons[0], lats[0]), (lons[-1], lats[-1])]:
        ax.plot([lon_end - px * ticklen, lon_end + px * ticklen],
                [lat_end - py * ticklen, lat_end + py * ticklen],
                color=color, lw=1.8, zorder=5)


def _add_north_arrow(ax, x=0.96, y=0.18, length=0.07):
    """North arrow in axis coordinates."""
    ax.annotate('', xy=(x, y + length), xytext=(x, y),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(x, y - 0.03, 'N', transform=ax.transAxes,
            ha='center', va='top', fontsize=9,
            color='white', fontweight='bold')


def _add_scale_bar(ax, ref_lat, x0_frac=0.65, y0_frac=0.05,
                   target_km=20):
    """
    Scale bar converting km to degrees longitude at ref_lat.
    target_km sets the approximate bar length in km.
    """
    R             = 6371.0
    km_per_deg    = R * np.cos(np.deg2rad(ref_lat)) * np.pi / 180.0
    bar_deg       = target_km / km_per_deg
    xlim, ylim    = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + x0_frac * (xlim[1] - xlim[0])
    y0 = ylim[0] + y0_frac * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + bar_deg], [y0, y0],
            color='white', lw=2.5, solid_capstyle='butt', zorder=5)
    for xpt in [x0, x0 + bar_deg]:
        ax.plot([xpt, xpt], [y0 - 0.005, y0 + 0.005],
                color='white', lw=1.5, zorder=5)
    ax.text(x0 + bar_deg / 2, y0 + 0.012, f'{target_km} km',
            ha='center', va='bottom', fontsize=8,
            color='white', fontweight='bold', zorder=5)


# ── MAIN FUNCTIONS ────────────────────────────────────────────────────────────

def extract_cross_section(ds_bathy, mooring_lon, mooring_lat,
                           trough_theta_deg,
                           transect_half_width_deg=0.5,
                           stop_depth_contour=None,
                           n_points=1000):
    """
    Extract a bathymetric cross-section perpendicular to the trough axis,
    passing through the mooring location.

    The mooring sits at the centre of the transect by construction.
    If stop_depth_contour is set the transect is trimmed on each side
    to stop where the seafloor first shoals above that depth, working
    outward from the mooring centre.

    Parameters
    ----------
    ds_bathy                 : xarray Dataset
        GEBCO bathymetry with 'elevation' variable on 'lon'/'lat' dims.
        Elevation is negative below sea level (standard GEBCO convention).
    mooring_lon, mooring_lat : float
        Mooring position in decimal degrees. xarray DataArrays are accepted
        and cast to float internally.
    trough_theta_deg         : float
        Trough axis angle CCW from East (degrees).
        Obtained from trough_orientation_from_moorings() in TNB_tools.
        The transect is drawn perpendicular to this angle.
    transect_half_width_deg  : float, default 0.5
        Half-width of the transect in degrees. Should be large enough to
        reach stop_depth_contour on both sides. At 75°S, 0.5° ~ 55 km.
    stop_depth_contour       : float or None
        Depth (m) at which to trim the transect on each side.
        e.g. 800 trims where seafloor shoals above 800 m.
        If None the full transect_half_width_deg extent is used.
    n_points                 : int, default 1000
        Number of sample points along the full transect before trimming.

    Returns
    -------
    dist_km        : ndarray, distance along transect from left end (km)
    depth_raw      : ndarray, seafloor depth (m, positive down)
    transect_lons  : ndarray, longitude of each transect point
    transect_lats  : ndarray, latitude of each transect point
    left_label     : str, compass direction at dist=0 (e.g. 'W')
    right_label    : str, compass direction at dist=max (e.g. 'E')

    Notes
    -----
    GEBCO does not resolve sub-ice bathymetry. If the transect crosses
    the Drygalski Ice Tongue the interpolated depth will reflect the ice
    surface elevation, not the seafloor. See module docstring for BedMachine
    alternative.
    """
    mooring_lon = float(mooring_lon)
    mooring_lat = float(mooring_lat)

    # Perpendicular to trough axis
    perp_theta = np.deg2rad(trough_theta_deg + 90.0)
    lat_scale  = np.cos(np.deg2rad(mooring_lat))

    t             = np.linspace(-transect_half_width_deg,
                                 transect_half_width_deg, n_points)
    transect_lons = mooring_lon + t * np.cos(perp_theta) / lat_scale
    transect_lats = mooring_lat + t * np.sin(perp_theta)

    # Crop bathymetry to transect extent then interpolate with scipy
    # (avoids xarray dimension conflicts with scalar coordinates)
    pad     = transect_half_width_deg + 0.1
    ds_crop = ds_bathy.sel(
        lon=slice(transect_lons.min() - pad, transect_lons.max() + pad),
        lat=slice(transect_lats.min() - pad, transect_lats.max() + pad)
    )
    interp_fn = RegularGridInterpolator(
        (ds_crop['lat'].values, ds_crop['lon'].values),
        ds_crop['elevation'].values,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    pts       = np.column_stack([transect_lats, transect_lons])
    depth_raw = -interp_fn(pts)   # flip sign: elevation → depth positive down

    # Distance array (flat-Earth, km)
    dlons_km = 6371.0 * lat_scale * np.deg2rad(np.diff(transect_lons))
    dlats_km = 6371.0 * np.deg2rad(np.diff(transect_lats))
    dist_km  = np.concatenate([[0], np.cumsum(np.sqrt(dlons_km**2 +
                                                        dlats_km**2))])

    # Trim to stop_depth_contour, searching outward from mooring at centre
    if stop_depth_contour is not None:
        mid = len(depth_raw) // 2

        left_idx = 0
        for i in range(mid, -1, -1):
            if depth_raw[i] < stop_depth_contour:
                left_idx = i
                break

        right_idx = len(depth_raw) - 1
        for i in range(mid, len(depth_raw)):
            if depth_raw[i] < stop_depth_contour:
                right_idx = i
                break

        print(f"Trimming to {stop_depth_contour} m contour:")
        print(f"  Left  edge: {dist_km[left_idx]:.2f} km, "
              f"depth = {depth_raw[left_idx]:.0f} m")
        print(f"  Right edge: {dist_km[right_idx]:.2f} km, "
              f"depth = {depth_raw[right_idx]:.0f} m")
        print(f"  Trimmed length: "
              f"{dist_km[right_idx] - dist_km[left_idx]:.1f} km")

        depth_raw     = depth_raw[left_idx:right_idx + 1]
        transect_lons = transect_lons[left_idx:right_idx + 1]
        transect_lats = transect_lats[left_idx:right_idx + 1]
        dist_km       = dist_km[left_idx:right_idx + 1]
        dist_km       = dist_km - dist_km[0]   # re-zero

    # Cardinal direction labels
    dlon        = transect_lons[-1] - transect_lons[0]
    dlat        = transect_lats[-1] - transect_lats[0]
    left_label  = _cardinal_label(-dlon, -dlat)
    right_label = _cardinal_label( dlon,  dlat)

    print(f"Transect: {left_label} (dist=0) → {right_label}")
    print(f"  Start : ({transect_lons[0]:.3f}°, {transect_lats[0]:.3f}°)")
    print(f"  End   : ({transect_lons[-1]:.3f}°, {transect_lats[-1]:.3f}°)")

    return dist_km, depth_raw, transect_lons, transect_lats, \
           left_label, right_label


def extract_cross_section_bedmachine(ds_bathy,
                                      mooring_lon, mooring_lat,
                                      trough_theta_deg,
                                      transect_half_width_deg=0.5,
                                      stop_depth_contour=None,
                                      n_points=1000,
                                      include_ice_cavity=True):
    """
    Extract a bathymetric cross-section from BedMachine Antarctica,
    perpendicular to the trough axis through the mooring location.

    Identical interface to extract_cross_section() but handles:
      - Polar stereographic input grid (EPSG:3031)
      - Sub-ice cavity bathymetry (mask == 3, floating ice)
      - Grounded ice / land masking (mask == 1 or 2 → NaN)

    Parameters
    ----------
    ds_bathy           : xarray Dataset, BedMachine TNB subset
    mooring_lon/lat    : float, mooring position (decimal degrees)
    trough_theta_deg   : float, trough axis angle CCW from East
    include_ice_cavity : bool, if True include sub-ice ocean beneath
                         floating ice (mask==3). If False, treat as land.
                         Default True — use the BedMachine advantage.
    """
    mooring_lon = float(mooring_lon)
    mooring_lat = float(mooring_lat)

    # ── Build transect in lon/lat then convert to PS ──────────────────────────
    perp_theta = np.deg2rad(trough_theta_deg + 90.0)
    lat_scale  = np.cos(np.deg2rad(mooring_lat))

    t             = np.linspace(-transect_half_width_deg,
                                 transect_half_width_deg, n_points)
    transect_lons = mooring_lon + t * np.cos(perp_theta) / lat_scale
    transect_lats = mooring_lat + t * np.sin(perp_theta)

    # Convert transect to polar stereographic
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3031',
                                        always_xy=True)
    transect_x, transect_y = transformer.transform(transect_lons,
                                                    transect_lats)

    # ── Interpolate bed and mask onto transect ────────────────────────────────
    x_grid = ds_bathy['x'].values.astype(float)
    y_grid = ds_bathy['y'].values.astype(float)
    bed    = ds_bathy['bed'].values.astype(float)
    mask   = ds_bathy['mask'].values.astype(float)

    # y may be descending — RegularGridInterpolator needs ascending
    if y_grid[0] > y_grid[-1]:
        y_grid = y_grid[::-1]
        bed    = bed[::-1, :]
        mask   = mask[::-1, :]

    interp_bed = RegularGridInterpolator(
        (y_grid, x_grid), bed,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    interp_mask = RegularGridInterpolator(
        (y_grid, x_grid), mask,
        method='nearest', bounds_error=False, fill_value=np.nan
    )

    pts        = np.column_stack([transect_y, transect_x])
    bed_vals   = interp_bed(pts)
    mask_vals  = interp_mask(pts)

    # ── Convert bed elevation to depth (positive down) ────────────────────────
    # bed is negative below sea level in BedMachine
    depth_raw = -bed_vals   # positive down

    # Mask land and grounded ice as NaN
    is_ocean = (mask_vals == 0) | (mask_vals == 3 if include_ice_cavity
                                    else False)
    depth_raw = np.where(is_ocean, depth_raw, np.nan)

    # ── Distance array (flat-Earth, km) ──────────────────────────────────────
    dlons_km = 6371.0 * lat_scale * np.deg2rad(np.diff(transect_lons))
    dlats_km = 6371.0 * np.deg2rad(np.diff(transect_lats))
    dist_km  = np.concatenate([[0], np.cumsum(
        np.sqrt(dlons_km**2 + dlats_km**2))])

    # ── Trim to stop_depth_contour ────────────────────────────────────────────
    if stop_depth_contour is not None:
        mid       = len(depth_raw) // 2
        left_idx  = 0
        right_idx = len(depth_raw) - 1

        for i in range(mid, -1, -1):
            d = depth_raw[i]
            if np.isnan(d) or d < stop_depth_contour:
                left_idx = i; break

        for i in range(mid, len(depth_raw)):
            d = depth_raw[i]
            if np.isnan(d) or d < stop_depth_contour:
                right_idx = i; break

        depth_raw     = depth_raw[left_idx:right_idx + 1]
        mask_vals     = mask_vals[left_idx:right_idx + 1]
        transect_lons = transect_lons[left_idx:right_idx + 1]
        transect_lats = transect_lats[left_idx:right_idx + 1]
        dist_km       = dist_km[left_idx:right_idx + 1]
        dist_km       = dist_km - dist_km[0]

        print(f"Trimmed to {stop_depth_contour}m contour: "
              f"{dist_km[-1]:.1f} km wide")

    # ── Cardinal labels ───────────────────────────────────────────────────────
    from trough_cross_section import _cardinal_label
    dlon        = transect_lons[-1] - transect_lons[0]
    dlat        = transect_lats[-1] - transect_lats[0]
    left_label  = _cardinal_label(-dlon, -dlat)
    right_label = _cardinal_label( dlon,  dlat)

    # Report ice cavity coverage
    n_cavity = np.sum(mask_vals == 3)
    if n_cavity > 0:
        cavity_km = n_cavity * np.mean(np.diff(dist_km))
        print(f"Sub-ice cavity spans {cavity_km:.1f} km of transect")
        print(f"  → BedMachine resolves seafloor beneath Drygalski Ice Tongue")

    print(f"Transect: {left_label} → {right_label}")

    return (dist_km, depth_raw, transect_lons, transect_lats,
            left_label, right_label, mask_vals)


def find_mooring_position_on_transect(dist_km, transect_lons,
                                       transect_lats,
                                       mooring_lon, mooring_lat):
    """
    Find where the mooring falls along the transect (km from left end).

    The mooring should be close to the midpoint since extract_cross_section
    centres the transect on it, but this finds the exact position by
    minimising geographic distance.

    Parameters
    ----------
    dist_km        : ndarray, distance array from extract_cross_section
    transect_lons  : ndarray, transect longitudes
    transect_lats  : ndarray, transect latitudes
    mooring_lon    : float or xarray DataArray
    mooring_lat    : float or xarray DataArray

    Returns
    -------
    float : distance along transect to mooring (km)
    """
    # Force plain floats — strips any attached xarray coordinates
    mooring_lon = float(mooring_lon)
    mooring_lat = float(mooring_lat)

    R         = 6371.0
    lat_scale = np.cos(np.deg2rad(mooring_lat))
    dx        = R * lat_scale * np.deg2rad(transect_lons - mooring_lon)
    dy        = R * np.deg2rad(transect_lats - mooring_lat)
    idx       = np.argmin(np.sqrt(dx**2 + dy**2))
    print(f"Mooring at {dist_km[idx]:.2f} km along transect "
          f"(midpoint = {dist_km[len(dist_km)//2]:.2f} km)")
    return dist_km[idx]


def compute_hssw_area(dist_km, depth,
                       hssw_thickness_m=500.0,
                       mooring_dist_km=None,
                       plot=True,
                       left_label='W',
                       right_label='E',
                       west_left=True):
    """
    Compute HSSW cross-sectional area using a fixed depth level for the
    top of the HSSW layer.

    The HSSW top is defined as the seafloor depth at the mooring minus
    hssw_thickness_m — a single constant depth level across the whole
    section. The layer thins naturally as trough walls shoal above this
    level, consistent with how a real water mass bounded by a density
    interface would behave.

    Parameters
    ----------
    dist_km          : ndarray, distance along transect (km)
    depth            : ndarray, seafloor depth (m, positive down)
    hssw_thickness_m : float, HSSW layer thickness at mooring location (m)
    mooring_dist_km  : float, mooring position along transect (km)
                       from find_mooring_position_on_transect().
                       Always pass this explicitly — the fallback (deepest
                       point) may not coincide with the mooring.
    plot             : bool, whether to generate cross-section figure
    left_label       : str, compass label for left end of plot (e.g. 'W')
    right_label      : str, compass label for right end (e.g. 'E')
    west_left        : bool, if True flip arrays so W is on left side

    Returns
    -------
    area_m2  : float, cross-sectional area of HSSW layer (m²)
    width_km : float, effective width of trough containing HSSW (km)
    results  : dict, intermediate arrays and scalars for further use
               keys: dist_km, depth, hssw_top_depth, hssw_layer,
                     contains_hssw, area_m2, width_km, mooring_depth
    """
    # Flip arrays so W is on the left if needed
    if west_left and right_label == 'W':
        dist_km         = dist_km[-1] - dist_km[::-1]
        depth           = depth[::-1]
        left_label, right_label = right_label, left_label
        if mooring_dist_km is not None:
            mooring_dist_km = dist_km[-1] - mooring_dist_km

    dist_m = dist_km * 1000.0   # km → m for area integration in m²

    # Seafloor depth at mooring
    if mooring_dist_km is None:
        mooring_idx = np.argmax(depth)
    else:
        mooring_idx = np.argmin(np.abs(dist_km - mooring_dist_km))
    mooring_depth  = depth[mooring_idx]

    # Fixed HSSW top depth (one horizontal level across the section)
    hssw_top_depth = mooring_depth - hssw_thickness_m

    print(f"Seafloor depth at mooring : {mooring_depth:.1f} m")
    print(f"HSSW top depth (fixed)    : {hssw_top_depth:.1f} m")
    print(f"HSSW thickness at mooring : {hssw_thickness_m:.1f} m")

    # Layer thickness at each transect point
    # Positive where seafloor is below hssw_top_depth, zero on walls
    hssw_layer    = np.maximum(depth - hssw_top_depth, 0.0)
    contains_hssw = depth > hssw_top_depth

    # Cross-sectional area (trapezoid integration, result in m²)
    area_m2  = np.trapezoid(hssw_layer, dist_m)
    width_km = np.sum(contains_hssw) * np.mean(np.diff(dist_km))

    print(f"\nEffective HSSW width  : {width_km:.1f} km")
    print(f"Cross-sectional area  : {area_m2/1e6:.3f} km²  "
          f"({area_m2:.3e} m²)")

    # Sensitivity table
    print("\n--- Sensitivity to HSSW thickness at mooring ---")
    for h in [300, 400, 500, 600, 700]:
        top   = mooring_depth - h
        layer = np.maximum(depth - top, 0.0)
        a     = np.trapezoid(layer, dist_m)
        print(f"  H = {h} m  →  top at {top:.0f} m  →  "
              f"A = {a/1e6:.3f} km²")

    if plot:
        fig, ax = plt.subplots(figsize=(11, 5))

        ax.fill_between(dist_km, depth, depth.max() * 1.05,
                        color='#8B7355', alpha=0.35, label='Seafloor')
        ax.plot(dist_km, depth, 'k-', lw=1.5, label='Bathymetry')

        hssw_bottom   = np.where(contains_hssw, depth,          np.nan)
        hssw_top_plot = np.where(contains_hssw, hssw_top_depth, np.nan)
        ax.fill_between(dist_km, hssw_top_plot, hssw_bottom,
                        where=contains_hssw, color='#534AB7', alpha=0.4,
                        label=f'HSSW layer (top at {hssw_top_depth:.0f} m)')
        ax.axhline(hssw_top_depth, color='#534AB7', lw=1.5, ls='--',
                   label=f'HSSW top = {hssw_top_depth:.0f} m')

        ax.axvline(dist_km[mooring_idx], color='#D85A30',
                   lw=1.5, ls=':', label='Mooring')
        ax.annotate(f'Mooring\n{mooring_depth:.0f} m',
                    xy=(dist_km[mooring_idx], mooring_depth),
                    xytext=(8, -20), textcoords='offset points',
                    color='#D85A30', fontsize=9)

        ylim_top = depth.max() * 1.05
        ax.text(dist_km[0],  ylim_top * 0.97, left_label,
                ha='left',  va='top', fontsize=11,
                fontweight='bold', color='#444441')
        ax.text(dist_km[-1], ylim_top * 0.97, right_label,
                ha='right', va='top', fontsize=11,
                fontweight='bold', color='#444441')
        ax.text(0.02, 0.08,
                f'A = {area_m2/1e6:.2f} km²\nWidth = {width_km:.1f} km',
                transform=ax.transAxes, fontsize=10, va='bottom',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.invert_yaxis()
        ax.set_xlabel('Distance along cross-section (km)', fontsize=11)
        ax.set_ylabel('Depth (m)', fontsize=11)
        ax.set_title('Drygalski Trough cross-section — HSSW layer area',
                     fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        plt.savefig('hssw_cross_section_area.png', dpi=150,
                    bbox_inches='tight')

    results = {
        'dist_km':        dist_km,
        'depth':          depth,
        'hssw_top_depth': hssw_top_depth,
        'hssw_layer':     hssw_layer,
        'contains_hssw':  contains_hssw,
        'area_m2':        area_m2,
        'width_km':       width_km,
        'mooring_depth':  mooring_depth,
    }
    return area_m2, width_km, results


def plot_bathy_with_transect(ds_bathy,
                              mooring_lon, mooring_lat,
                              transect_lons, transect_lats,
                              mooring_lon2=None, mooring_lat2=None,
                              transect_lons2=None, transect_lats2=None,
                              mooring_lon3=None, mooring_lat3=None,
                              transect_lons3=None, transect_lats3=None,
                              lon_bounds=None, lat_bounds=None,
                              depth_contours=(500, 800, 1100, 1400),
                              save_path=None):
    """
    Plan-view bathymetry map with mooring locations and transect lines.

    Accepts up to three moorings, each with an optional transect. Transect
    lines are colour-coded to match their mooring marker. End ticks on each
    transect line indicate the finite cross-section face.

    Parameters
    ----------
    ds_bathy                        : xarray Dataset, GEBCO bathymetry
    mooring_lon, mooring_lat        : float, primary mooring (M1)
    transect_lons, transect_lats    : ndarray, M1 transect coordinates
    mooring_lon2/lat2               : float, optional second mooring (M2)
    transect_lons2/lats2            : ndarray, optional M2 transect
    mooring_lon3/lat3               : float, optional third mooring (M3)
    transect_lons3/lats3            : ndarray, optional M3 transect
    lon_bounds                      : (min, max) longitude crop in 0-360°
                                      for the Ross Sea. Defaults to transect
                                      extent + 0.3° padding.
    lat_bounds                      : (min, max) latitude crop
    depth_contours                  : tuple of depth levels (m) for isobaths
    save_path                       : output filename

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Normalise coordinate/variable names (GEBCO uses various conventions)
    ds = ds_bathy.copy()
    for name in ['lon', 'longitude', 'x', 'nav_lon']:
        if name in ds.dims or name in ds.coords:
            ds = ds.rename({name: 'lon'}); break
    for name in ['lat', 'latitude', 'y', 'nav_lat']:
        if name in ds.dims or name in ds.coords:
            ds = ds.rename({name: 'lat'}); break
    for name in ['elevation', 'z', 'depth', 'topo', 'Band1']:
        if name in ds.data_vars:
            ds = ds.rename({name: 'elevation'}); break

    mooring_lon = float(mooring_lon)
    mooring_lat = float(mooring_lat)

    # Auto-bounds spanning all transects
    all_lons = [transect_lons]
    all_lats = [transect_lats]
    if transect_lons2 is not None:
        all_lons.append(transect_lons2); all_lats.append(transect_lats2)
    if transect_lons3 is not None:
        all_lons.append(transect_lons3); all_lats.append(transect_lats3)
    all_lons = np.concatenate(all_lons)
    all_lats = np.concatenate(all_lats)

    if lon_bounds is None:
        pad = 0.3
        lon_bounds = (all_lons.min() - pad, all_lons.max() + pad)
    if lat_bounds is None:
        pad = 0.3
        lat_bounds = (all_lats.min() - pad, all_lats.max() + pad)

    # Handle ascending or descending lat storage
    lat_vals = ds.lat.values
    if lat_vals[0] > lat_vals[-1]:
        ds_crop = ds.sel(lon=slice(*lon_bounds),
                         lat=slice(lat_bounds[1], lat_bounds[0]))
    else:
        ds_crop = ds.sel(lon=slice(*lon_bounds),
                         lat=slice(*lat_bounds))

    if ds_crop['elevation'].size == 0:
        raise ValueError(
            "Crop returned empty array. Pass lon_bounds and lat_bounds "
            "explicitly in 0-360° format for the Ross Sea.\n"
            f"Try: lon_bounds=({mooring_lon - 1:.1f}, "
            f"{mooring_lon + 1:.1f}), "
            f"lat_bounds=({mooring_lat - 1:.1f}, {mooring_lat + 1:.1f})")

    depth = -ds_crop['elevation'].values
    lons  =  ds_crop['lon'].values
    lats  =  ds_crop['lat'].values
    LON, LAT = np.meshgrid(lons, lats)

    depth_ocean = depth[depth > 0]
    if depth_ocean.size == 0:
        raise ValueError("No ocean values found in cropped region.")
    vmax       = np.nanpercentile(depth_ocean, 99)
    depth_plot = np.where(depth > 0, depth, np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))

    pcm = ax.pcolormesh(LON, LAT, depth_plot, cmap='Blues',
                         shading='auto', vmin=0, vmax=vmax)
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label('Depth (m)', fontsize=10)

    land = np.where(depth <= 0, 1.0, np.nan)
    ax.pcolormesh(LON, LAT, land, cmap='Greys', shading='auto',
                   vmin=0, vmax=1, alpha=0.6)

    valid_contours = [c for c in depth_contours
                      if c > depth_ocean.min() and c < vmax]
    if valid_contours:
        cs = ax.contour(LON, LAT, depth_plot, levels=valid_contours,
                         colors='white', linewidths=0.7, alpha=0.6)
        ax.clabel(cs, fmt='%dm', fontsize=7, colors='white', inline=True)

    # Transect lines (colour-matched to mooring markers)
    c1, c2, c3 = '#D85A30', '#F5C4B3', '#9FE1CB'

    ax.plot(transect_lons, transect_lats, color=c1, lw=2.0,
            zorder=4, label='Transect M1')
    _add_transect_endticks(ax, transect_lons, transect_lats, color=c1)

    if transect_lons2 is not None:
        ax.plot(transect_lons2, transect_lats2, color=c2, lw=2.0,
                zorder=4, label='Transect M2')
        _add_transect_endticks(ax, transect_lons2, transect_lats2, color=c2)

    if transect_lons3 is not None:
        ax.plot(transect_lons3, transect_lats3, color=c3, lw=2.0,
                zorder=4, label='Transect M3')
        _add_transect_endticks(ax, transect_lons3, transect_lats3, color=c3)

    # Mooring markers
    moor_kw = dict(marker='v', markersize=9, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.8,
                    linestyle='none')
    ax.plot(mooring_lon, mooring_lat, color=c1, label='M1', **moor_kw)
    ax.annotate('M1', (mooring_lon, mooring_lat), xytext=(5, 6),
                textcoords='offset points', fontsize=9,
                color='white', fontweight='bold', zorder=7)

    if mooring_lon2 is not None:
        ax.plot(float(mooring_lon2), float(mooring_lat2),
                color=c2, label='M2', **moor_kw)
        ax.annotate('M2', (float(mooring_lon2), float(mooring_lat2)),
                    xytext=(5, 6), textcoords='offset points',
                    fontsize=9, color='white', fontweight='bold', zorder=7)

    if mooring_lon3 is not None:
        ax.plot(float(mooring_lon3), float(mooring_lat3),
                color=c3, label='M3', **moor_kw)
        ax.annotate('M3', (float(mooring_lon3), float(mooring_lat3)),
                    xytext=(5, 6), textcoords='offset points',
                    fontsize=9, color='white', fontweight='bold', zorder=7)

    _add_north_arrow(ax)
    _add_scale_bar(ax, mooring_lat)

    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title('Drygalski Trough — bathymetry and cross-section transects',
                 fontsize=11)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.2f}°'))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f'{y:.2f}°'))
    ax.legend(loc='upper left', fontsize=8,
              framealpha=0.7, edgecolor='gray')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# import xarray as xr
# from trough_cross_section import (
#     extract_cross_section,
#     find_mooring_position_on_transect,
#     compute_hssw_area,
#     plot_bathy_with_transect,
# )
#
# # ── Load bathymetry ────────────────────────────────────────────────────────
# bath = xr.open_dataset('gebco_2023_TNB.nc')
#
# # ── Trough axis angle from your mooring positions (from TNB_tools) ─────────
# # theta, _, _, _ = trough_orientation_from_moorings(lon1, lat1, lon2, lat2)
# theta = 170.0   # example: trough runs roughly N-S with slight tilt
#
# # ── M1 cross-section ───────────────────────────────────────────────────────
# dist1, depth1, t_lons1, t_lats1, left1, right1 = extract_cross_section(
#     ds_bathy                = bath,
#     mooring_lon             = DITD_ctd_top.lon,
#     mooring_lat             = DITD_ctd_top.lat,
#     trough_theta_deg        = theta,
#     transect_half_width_deg = 0.5,    # large enough to reach stop contour
#     stop_depth_contour      = 800,    # trim where seafloor shoals < 800 m
# )
#
# moor_dist1 = find_mooring_position_on_transect(
#     dist1, t_lons1, t_lats1,
#     DITD_ctd_top.lon, DITD_ctd_top.lat
# )
#
# area1, width1, results1 = compute_hssw_area(
#     dist1, depth1,
#     hssw_thickness_m = 500.0,
#     mooring_dist_km  = moor_dist1,
#     left_label       = left1,
#     right_label      = right1,
#     west_left        = True,
# )
#
# # ── M2 cross-section (repeat for second along-trough mooring) ─────────────
# dist2, depth2, t_lons2, t_lats2, left2, right2 = extract_cross_section(
#     ds_bathy                = bath,
#     mooring_lon             = M2.lon,
#     mooring_lat             = M2.lat,
#     trough_theta_deg        = theta,
#     transect_half_width_deg = 0.5,
#     stop_depth_contour      = 800,
# )
#
# moor_dist2 = find_mooring_position_on_transect(
#     dist2, t_lons2, t_lats2, M2.lon, M2.lat
# )
#
# area2, width2, results2 = compute_hssw_area(
#     dist2, depth2,
#     hssw_thickness_m = 500.0,
#     mooring_dist_km  = moor_dist2,
#     left_label       = left2,
#     right_label      = right2,
#     west_left        = True,
# )
#
# # ── Plan view map with all three moorings and their transects ──────────────
# fig, ax = plot_bathy_with_transect(
#     ds_bathy         = bath,
#     mooring_lon      = DITD_ctd_top.lon,
#     mooring_lat      = DITD_ctd_top.lat,
#     transect_lons    = t_lons1,
#     transect_lats    = t_lats1,
#     mooring_lon2     = M2.lon,
#     mooring_lat2     = M2.lat,
#     transect_lons2   = t_lons2,
#     transect_lats2   = t_lats2,
#     mooring_lon3     = M3.lon,          # western mooring (no transect yet)
#     mooring_lat3     = M3.lat,
#     lon_bounds       = (163.0, 165.5),  # 0-360° format for Ross Sea
#     lat_bounds       = (-76.0, -74.5),
#     depth_contours   = (500, 800, 1100, 1400),
#     save_path        = 'plan_view_transects.png',
# )
#
# # ── Use area in volume transport estimate ──────────────────────────────────
# # volume_transport_Sv = v_along_lowpassed * area1 / 1e6
# #
# # NOTE on BedMachine: if the transect crosses the Drygalski Ice Tongue,
# # GEBCO will show a land barrier where ocean exists beneath the ice.
# # Replace extract_cross_section with a BedMachine equivalent that reads
# # the 'bed' variable (EPSG:3031 polar stereographic) and uses the 'mask'
# # variable to identify floating ice (mask == 3). Reproject transect
# # coordinates from lon/lat to polar stereographic with pyproj before
# # calling RegularGridInterpolator.