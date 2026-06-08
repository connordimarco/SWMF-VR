"""CAP 2 -- seeding points from the boundary.

Convert per-MLT equatorial boundary radii (r, MLT) into field-line SEED points:
(r, MLT) -> SM equatorial (x, y, 0) -> GSM via geopack.smgsm. These seeds feed
the Stage B StreamTracer.

CONTRACT (preserved from v6, parity depends on it):
  * geopack.recalc(ut) MUST be called for this timestep BEFORE write_seed_csv
    (the caller does it once per frame; see swmf_vr.coords.recalc_for). smgsm
    reads geopack's recalc'd global dipole state.
  * The header is ALWAYS written, even with zero boundary rows, so a no-boundary
    frame yields a header-only CSV that Stage B skips.
  * Number formatting is exact: %.4f for coords/r/MLT, %.6e for flux values.
"""

import numpy as np
from geopack import geopack


def write_seed_csv(r_arr, mlt_centers, flux_label, out_path, flux_peak=None):
    """Write equatorial boundary points as GSM field-line seed points.

    r_arr        per-MLT boundary radius (NaN where none) -- only finite kept.
    mlt_centers  equatorial MLT per column.
    flux_label   flux value to record as `Flux` -- a scalar (threshold) or a
                 per-MLT array (peak flux for the 'max'/danger surface).
    flux_peak    per-MLT column-peak flux for `Flux_peak` (varies on every
                 surface). If None or non-finite for a column, reuses Flux.
    """
    valid = np.where(np.isfinite(r_arr))[0]
    rows = []
    for m in valid:
        r = float(r_arr[m])
        mlt_eq = float(mlt_centers[m])
        phi_rad = np.radians((mlt_eq - 12.0) * 15.0)
        x_sm = r * np.cos(phi_rad)
        y_sm = r * np.sin(phi_rad)
        x_gsm, y_gsm, _ = geopack.smgsm(float(x_sm), float(y_sm), 0.0, 1)
        if np.ndim(flux_label) > 0:
            flux_val = float(flux_label[m])
        else:
            flux_val = float(flux_label)
        if flux_peak is not None and np.isfinite(flux_peak[m]):
            peak_val = float(flux_peak[m])
        else:
            peak_val = flux_val
        rows.append((x_gsm, y_gsm, r, mlt_eq, flux_val, peak_val))

    with open(out_path, 'w') as f:
        f.write('Seed_X_Re,Seed_Y_Re,Seed_Z_Re,Equator_R,Equator_MLT,Flux,Flux_peak\n')
        for x, y, r, mlt_eq, flux_val, peak_val in rows:
            f.write(f'{x:.4f},{y:.4f},0.0000,{r:.4f},{mlt_eq:.4f},'
                    f'{flux_val:.6e},{peak_val:.6e}\n')
