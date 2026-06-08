"""Find the belt boundaries in an equatorial flux slice and turn them into
GSM field-line seed points.

Boundaries are per-MLT radii: a threshold crossing (inward = outer edge,
outward = inner edge) or the radius of the per-MLT flux peak. Seeds convert
(r, MLT) -> SM equatorial -> GSM via geopack; recalc(ut) must be set for the
frame first (frames.recalc_for).
"""

import numpy as np
from geopack import geopack


def interpolate_crossing(threshold, f1, f2, r1, r2):
    if f2 == f1:
        return r1
    return r1 + (threshold - f1) / (f2 - f1) * (r2 - r1)


def find_boundary(flux_slice, ro_vals, threshold, direction):
    """Per-MLT radius where flux crosses threshold. direction 'inward' walks
    from large r (outer edge), 'outward' from small r (inner edge)."""
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
            a, b = order[j], order[j + 1]
            if col_flux[a] < threshold and col_flux[b] >= threshold:
                result_r[m] = interpolate_crossing(
                    threshold, col_flux[a], col_flux[b], col_ro[a], col_ro[b])
                break
    return result_r


def find_max_boundary(flux_slice, ro_vals):
    """Per-MLT radius of peak flux (the danger surface)."""
    num_lats, num_mlts = flux_slice.shape
    result_r = np.full(num_mlts, np.nan)
    for m in range(num_mlts):
        col_flux = flux_slice[:, m]
        col_ro = ro_vals[:, m]
        valid = np.where((col_ro >= 1.0) & np.isfinite(col_ro) & np.isfinite(col_flux))[0]
        if len(valid) == 0:
            continue
        result_r[m] = col_ro[valid[int(np.nanargmax(col_flux[valid]))]]
    return result_r


def find_peak_flux(flux_slice, ro_vals):
    """Per-MLT peak flux value (matches find_max_boundary's radii)."""
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


def write_seed_csv(r_arr, mlt_centers, flux_label, out_path, flux_peak=None):
    """Write boundary radii as GSM seed points. Always writes the header, so a
    no-boundary frame leaves a header-only file (trace skips those).

    flux_label is the Flux column: a scalar (threshold) for inner/outer, or a
    per-MLT array (the peak) for danger. flux_peak is the always-varying column
    peak written as Flux_peak.
    """
    valid = np.where(np.isfinite(r_arr))[0]
    rows = []
    for m in valid:
        r = float(r_arr[m])
        mlt_eq = float(mlt_centers[m])
        phi = np.radians((mlt_eq - 12.0) * 15.0)
        x_gsm, y_gsm, _ = geopack.smgsm(r * np.cos(phi), r * np.sin(phi), 0.0, 1)
        flux_val = float(flux_label[m]) if np.ndim(flux_label) > 0 else float(flux_label)
        peak_val = float(flux_peak[m]) if flux_peak is not None and np.isfinite(flux_peak[m]) else flux_val
        rows.append((x_gsm, y_gsm, r, mlt_eq, flux_val, peak_val))

    with open(out_path, 'w') as f:
        f.write('Seed_X_Re,Seed_Y_Re,Seed_Z_Re,Equator_R,Equator_MLT,Flux,Flux_peak\n')
        for x, y, r, mlt_eq, flux_val, peak_val in rows:
            f.write(f'{x:.4f},{y:.4f},0.0000,{r:.4f},{mlt_eq:.4f},'
                    f'{flux_val:.6e},{peak_val:.6e}\n')
