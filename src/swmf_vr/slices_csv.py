"""CSV export of the meridional flux slices about the dipole axis.

Shared by scripts/fp_slices.py (computes the slices under pvbatch and writes
the CSV in the same run) and scripts/fp_slices_csv.py (re-exports the CSV from
the saved npz cubes under plain python3, so a change of CSV layout never needs
the 91 s/frame BATSRUS sampling again).

Layout (2026-09-16, after the VR team's first look):
  * default: ONE file per timestamp holding all azimuth cuts, the cut being the
    `azimuth_deg` column (the VR side consumes XYZ point lists, so the cuts are
    just a sortable column);
  * `bands`: which energy bands to write as flux columns (default all three);
  * `drop_zeros`: leave out rows where every written flux column is 0. Zero
    means "no particles here" (outside the belt, open line, off the RBE grid,
    inside Earth), so on a point list nothing is lost; the regular grid is
    stated in the header for anyone who wants it dense again.
"""

import os
import json
import math

import numpy as np
from geopack import geopack

BAND_KEYS = ['lt1', '1to2', 'gt2']                    # cube band index 0, 1, 2
BAND_COLUMNS = {'lt1': 'flux_lt1MeV', '1to2': 'flux_1to2MeV', 'gt2': 'flux_gt2MeV'}
BAND_LABELS = {'lt1': '<1 MeV', '1to2': '1-2 MeV', 'gt2': '>2 MeV'}


def parse_bands(spec):
    """'gt2' / 'lt1,gt2' / 'all' -> ordered list of band keys."""
    if spec is None or spec == 'all':
        return list(BAND_KEYS)
    keys = [s.strip() for s in spec.split(',') if s.strip()]
    bad = [k for k in keys if k not in BAND_KEYS]
    if bad or not keys:
        raise ValueError('bands must be a comma list of {} (or all), got {!r}'
                         .format('/'.join(BAND_KEYS), spec))
    return [k for k in BAND_KEYS if k in keys]


def sm_to_gsm_matrix():
    """3x3 R with R @ v_sm = v_gsm. geopack.recalc must be set for the frame."""
    return np.column_stack([np.array(geopack.smgsm(1.0, 0.0, 0.0, 1)),
                            np.array(geopack.smgsm(0.0, 1.0, 0.0, 1)),
                            np.array(geopack.smgsm(0.0, 0.0, 1.0, 1))])


def sm_to_frame_matrix(frame):
    """3x3 R with R @ v_sm = v_<frame>, for frame in {sm, gsm, gei}."""
    if frame == 'sm':
        return np.eye(3)
    if frame == 'gsm':
        return sm_to_gsm_matrix()
    if frame == 'gei':
        def sm2gei(x, y, z):
            gx, gy, gz = geopack.smgsm(x, y, z, 1)      # SM -> GSM
            gx, gy, gz = geopack.geogsm(gx, gy, gz, -1)  # GSM -> GEO
            return geopack.geigeo(gx, gy, gz, -1)        # GEO -> GEI
        return np.column_stack([np.array(sm2gei(1.0, 0.0, 0.0)),
                                np.array(sm2gei(0.0, 1.0, 0.0)),
                                np.array(sm2gei(0.0, 0.0, 1.0))])
    raise ValueError('frame must be sm|gsm|gei, got {!r}'.format(frame))


def build_slice_points_sm(azimuths_deg, rho, z):
    """SM points for every slice. Returns (pts_sm (N,3), n_rho, n_z); the per-
    slice image index of point k is (k // (n_rho*n_z))=slice, then (iz, irho)
    row-major within z x rho."""
    n_rho, n_z = len(rho), len(z)
    RHO, Z = np.meshgrid(rho, z)                      # [iz, irho]
    blocks = []
    for az in azimuths_deg:
        phi = math.radians(az)
        xs = (RHO * math.cos(phi)).ravel()
        ys = (RHO * math.sin(phi)).ravel()
        zs = Z.ravel()
        blocks.append(np.column_stack([xs, ys, zs]))
    return np.vstack(blocks), n_rho, n_z


def slices_to_rows(azimuths_deg, rho, z, cube, R_out, bands, drop_zeros):
    """Flatten (n_az, n_z, n_rho, 3) cube to a row table
    [azimuth, rho, z, x, y, z_out, flux...] in the output frame.
    Returns (rows (N, 6+len(bands)), n_kept, n_total)."""
    band_idx = [BAND_KEYS.index(b) for b in bands]
    pts_sm, n_rho, n_z = build_slice_points_sm(azimuths_deg, rho, z)
    pts_out = (np.asarray(R_out) @ pts_sm.T).T
    per_slice = n_rho * n_z
    az_col = np.repeat(np.asarray(azimuths_deg, dtype=float), per_slice)
    RHO, Z = np.meshgrid(rho, z)
    rho_col = np.tile(RHO.ravel(), len(azimuths_deg))
    z_col = np.tile(Z.ravel(), len(azimuths_deg))
    flux = np.asarray(cube, dtype=float)[..., band_idx].reshape(-1, len(band_idx))
    rows = np.column_stack([az_col, rho_col, z_col, pts_out, flux])
    n_total = len(rows)
    if drop_zeros:
        rows = rows[(flux > 0).any(axis=1)]
    return rows, len(rows), n_total


def write_slices_csv(path, azimuths_deg, rho, z, cube, R_out, frame, ts, quantity,
                     bands=None, drop_zeros=False, integer_flux=False):
    """Write one CSV holding the given azimuth cuts. `cube` is
    (len(azimuths_deg), n_z, n_rho, 3), already finite (no NaN). Returns
    (n_rows_written, n_rows_total).

    `integer_flux=True` writes every flux value as a plain rounded integer.
    The default '%.5g' switches to exponent notation ('1.5496e+05') at exactly
    1e5, which is also about the level of the outer shell, so a reader that
    cannot parse the e+NN form drops precisely the inside of the belt
    (2026-10-05, Mike Makivic's importer). With drop_zeros the zero test is
    applied after rounding, so no all-zero row is written."""
    bands = list(BAND_KEYS) if bands is None else list(bands)
    rows, n_kept, n_total = slices_to_rows(azimuths_deg, rho, z, cube, R_out, bands, drop_zeros)
    if integer_flux:
        rows = np.array(rows, dtype=float, copy=True)
        rows[:, 6:] = np.rint(rows[:, 6:])
        if drop_zeros and len(rows):
            rows = rows[np.any(rows[:, 6:] != 0.0, axis=1)]
            n_kept = len(rows)
    step = float(rho[1] - rho[0]) if len(rho) > 1 else float('nan')
    unit = 'particles/(cm^2 s)' if quantity == 'omni' else 'particles/(cm^2 s sr)'
    az_list = ','.join('{:g}'.format(a) for a in azimuths_deg)
    header = [
        '# SWMF-VR flux slices  ts={}  {} meridional half-plane cut(s) about the SM (dipole) axis'
        .format(ts, len(azimuths_deg)),
        '# azimuth_deg = {}  (0 = noon, MLT = azimuth/15 + 12; one cut per value, filter on this column)'
        .format(az_list),
        '# in-plane grid per cut: rho 0..{:g} Re, z {:g}..{:g} Re, step {:g} Re; x/y/z positions in {} (Re)'
        .format(rho[-1], z[0], z[-1], step, frame.upper()),
        '# flux columns: {}; {}. No NaNs. {}'.format(
            ', '.join('{}={}'.format(BAND_COLUMNS[b], BAND_LABELS[b]) for b in bands), unit,
            'Rows with all flux columns = 0 (no particles) omitted: {} of {} grid points kept.'
            .format(n_kept, n_total) if drop_zeros else 'Empty grid points (no particles) written as 0.'),
    ]
    if integer_flux:
        header.append('# flux values are plain rounded integers (no exponent notation)')
    cols = ['azimuth_deg', 'rho_re', 'z_re', 'x_{f}'.format(f=frame),
            'y_{f}'.format(f=frame), 'z_{f}'.format(f=frame)] + [BAND_COLUMNS[b] for b in bands]
    fmt = ['%.1f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f'] + \
          ['%.0f' if integer_flux else '%.5g'] * len(bands)
    with open(path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        f.write(','.join(cols) + '\n')
        np.savetxt(f, rows, fmt=fmt, delimiter=',')
    return n_kept, n_total



def gei_static_bbox(timestamps, rho_max, z_max, frame='gei'):
    """Half-widths [sx, sy, sz] of the smallest axis-aligned box in `frame`
    that contains the sampling cylinder (rho <= rho_max, |z| <= z_max about the
    SM/dipole axis) for EVERY frame in `timestamps`. The cylinder is symmetric
    about the origin, so the box is [-s, s] per axis. The support of the
    cylinder along a unit direction e is rho_max*hypot(a_x, a_y) + z_max*|a_z|
    with a = R^T e (a is frame-axis i expressed in SM = row i of R)."""
    import datetime
    from . import frames as _frames
    s = np.zeros(3)
    for ts in timestamps:
        _frames.recalc_for(datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S'))
        R = sm_to_frame_matrix(frame)             # R @ v_sm = v_frame
        for i in range(3):
            a = R[i]                              # frame axis i, in SM coords
            s[i] = max(s[i], rho_max * math.hypot(a[0], a[1]) + z_max * abs(a[2]))
    return [round(float(v), 4) for v in s]


def write_grid_json(path, azimuths_deg, rho, z, frame, quantity, bands=None,
                    drop_zeros=False, axis='SM_dipole', bbox_xyz=None):
    """Sidecar describing the (static) sampling grid the slice CSVs live on, so
    the VR side can allocate and index a 3D texture. The grid is a cylinder in
    the SM (dipole-axis) frame; it is IDENTICAL for every frame. The CSV x/y/z
    columns are this cylinder rigidly rotated into `frame` (GEI) per timestamp,
    so the stable, frame-independent extents are the cylindrical ones below."""
    bands = list(BAND_KEYS) if bands is None else list(bands)
    rho = np.asarray(rho, dtype=float); z = np.asarray(z, dtype=float)
    az = np.asarray(azimuths_deg, dtype=float)
    step = round(float(rho[1] - rho[0]), 6) if len(rho) > 1 else None
    az_step = round(float(az[1] - az[0]), 6) if len(az) > 1 else None
    r_env = round(float(math.hypot(rho.max(), max(abs(z.min()), abs(z.max())))), 4)
    doc = {
        'units': 'Re (Earth radii; 1 Re = 6371.2 km, geopack/SWMF convention)',
        'coordinate_system': 'cylindrical about the SM (dipole) axis',
        'axis': axis,
        'grid_is_static_across_frames': True,
        'position_frame': frame.upper(),
        'position_note': ('CSV x_{f}/y_{f}/z_{f} are the cylinder below rotated '
                          'rigidly into {F} per frame (dipole tilt ~15-22 deg); '
                          'build the texture in (rho, phi, z) and it is identical '
                          'every frame, then apply the one per-frame rotation to '
                          'place it.').format(f=frame.lower(), F=frame.upper()),
        'rho_re':      {'min': round(float(rho.min()), 6), 'max': round(float(rho.max()), 6),
                        'n': int(len(rho)), 'step': step},
        'z_re':        {'min': round(float(z.min()), 6), 'max': round(float(z.max()), 6),
                        'n': int(len(z)), 'step': step},
        'azimuth_deg': {'min': float(az.min()), 'max': float(az.max()),
                        'n': int(len(az)), 'step': az_step,
                        'periodic_mod_360': True, 'convention': '0 = noon, MLT = azimuth/15 + 12',
                        'values': az.tolist()},
        'bounding_box_re': {
            'frame': frame.upper(),
            'note': ('axis-aligned XYZ box to allocate the 3D texture; the '
                     'x_{f}/y_{f}/z_{f} columns fall inside it. It is symmetric '
                     'about the origin (Earth center).').format(f=frame.lower()),
            'x': {'min': (-bbox_xyz[0] if bbox_xyz else -r_env),
                  'max': (bbox_xyz[0] if bbox_xyz else r_env)},
            'y': {'min': (-bbox_xyz[1] if bbox_xyz else -r_env),
                  'max': (bbox_xyz[1] if bbox_xyz else r_env)},
            'z': {'min': (-bbox_xyz[2] if bbox_xyz else -r_env),
                  'max': (bbox_xyz[2] if bbox_xyz else r_env)},
            'kind': ('static: contains every frame of the run (tight, computed '
                     'over all frames)' if bbox_xyz else
                     'loose static envelope = sqrt(rho_max^2 + |z|_max^2), '
                     'contains the cylinder for any rotation')},
        'flux_columns': [BAND_COLUMNS[b] for b in bands],
        'flux_labels': [BAND_LABELS[b] for b in bands],
        'flux_unit': 'particles/(cm^2 s)' if quantity == 'omni' else 'particles/(cm^2 s sr)',
        'fill': ('rows are one grid point each; missing rows (dropped because '
                 'flux = 0) are outside the belt / open line / off the RBE grid '
                 'and should be filled with 0' if drop_zeros else
                 'every grid point present; empty points written as 0'),
        'files': 'slices_<YYYYMMDD_HHMMSS>.csv, one per frame, all azimuth cuts stacked (azimuth_deg column)',
    }
    tmp = path + '.tmp.{}'.format(os.getpid())
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)                     # atomic; safe under the sharded array
    return doc
