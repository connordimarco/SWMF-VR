"""CAP 1 -- defining a boundary.

Per-MLT radial boundary extraction from an equatorial flux slice. Pure numpy;
verbatim port of the v6 plot_flux_slices functions so results are bit-identical.

  find_boundary       threshold-crossing search (inward = outer edge,
                      outward = inner edge), linearly interpolated.
  find_max_boundary   radius of the per-MLT flux peak (the 'danger' surface).
  find_peak_flux      the per-MLT peak flux value (color scalar Flux_peak).
"""

import numpy as np


def interpolate_crossing(threshold, f1, f2, r1, r2):
    """Linear interpolant for the radius where flux crosses `threshold`."""
    if f2 == f1:
        return r1
    t = (threshold - f1) / (f2 - f1)
    return r1 + t * (r2 - r1)


def find_boundary(flux_slice, ro_vals, threshold, direction):
    """Find boundary at threshold by searching in direction ('inward'/'outward').

    inward:  start at largest r, walk toward Earth -- outer boundary.
    outward: start at smallest r, walk away from Earth -- inner boundary.
    Returns an array of radii per MLT column (NaN where no crossing).
    """
    num_lats, num_mlts = flux_slice.shape
    result_r = np.full(num_mlts, np.nan)

    for m in range(num_mlts):
        col_flux = flux_slice[:, m]
        col_ro = ro_vals[:, m]

        valid = np.where((col_ro >= 1.0) & np.isfinite(col_ro) & np.isfinite(col_flux))[0]
        if len(valid) < 2:
            continue

        if direction == 'inward':
            order = valid[np.argsort(col_ro[valid])[::-1]]
        else:
            order = valid[np.argsort(col_ro[valid])]

        for j in range(len(order) - 1):
            idx_a = order[j]
            idx_b = order[j + 1]
            if col_flux[idx_a] < threshold and col_flux[idx_b] >= threshold:
                result_r[m] = interpolate_crossing(
                    threshold, col_flux[idx_a], col_flux[idx_b],
                    col_ro[idx_a], col_ro[idx_b])
                break

    return result_r


def find_max_boundary(flux_slice, ro_vals):
    """Per-MLT radius of peak flux (the 'danger'/max surface)."""
    num_lats, num_mlts = flux_slice.shape
    result_r = np.full(num_mlts, np.nan)

    for m in range(num_mlts):
        col_flux = flux_slice[:, m]
        col_ro = ro_vals[:, m]

        valid = np.where((col_ro >= 1.0) & np.isfinite(col_ro) & np.isfinite(col_flux))[0]
        if len(valid) == 0:
            continue

        peak_idx = valid[int(np.nanargmax(col_flux[valid]))]
        result_r[m] = col_ro[peak_idx]

    return result_r


def find_peak_flux(flux_slice, ro_vals):
    """Per-MLT peak flux value (aligned with find_max_boundary's radii)."""
    num_lats, num_mlts = flux_slice.shape
    result_f = np.full(num_mlts, np.nan)
    for m in range(num_mlts):
        col_flux = flux_slice[:, m]
        col_ro = ro_vals[:, m]
        valid = np.where((col_ro >= 1.0) & np.isfinite(col_ro) & np.isfinite(col_flux))[0]
        if len(valid) == 0:
            continue
        result_f[m] = np.nanmax(col_flux[valid])
    return result_f
