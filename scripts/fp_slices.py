#!/usr/bin/env python3
"""Meridional slices of the foot-point flux, every 15 deg around the DIPOLE
(SM) axis, with NO NaNs (pvbatch).

  "$PV_BATCH" scripts/fp_slices.py --ts 20240511_094100
       [--step RE] [--extent RE] [--n-slices N] [--quantity omni|perp]
       [--out-root DIR] [--frame gei|sm|gsm] [--force] [--no-plot]
       [--bands all|gt2|lt1,1to2,...] [--drop-zeros] [--split-azimuths]

CSV layout (2026-09-16, after the VR team's first look at a frame): one file
per timestamp with all cuts stacked, the cut being the `azimuth_deg` column
(--split-azimuths restores one file per cut); --bands picks the flux columns;
--drop-zeros omits rows where every written flux is 0. The npz cube is always
dense and complete; scripts/fp_slices_csv.py re-exports any CSV layout from it.

Why slices, and why the dipole axis
-----------------------------------
The VR team wants the full 3D flux back on the model, but reconstructed from
2D slices they can interpolate between, not from the old gappy Cartesian
voxel box. We sample the flux DIRECTLY on each slice plane (as fp_cutaway.py
shades its single y=0 face), so there is no coarse-grid -> plane resampling.

The slices are meridional HALF-planes in SM (Solar Magnetic) coordinates,
rotated in 15 deg azimuth steps about the dipole axis (SM z). The belt is
organised around the magnetic axis, so it is most axisymmetric in magnetic
local time -- meaning adjacent slices differ least and the between-slice
linear interpolation the VR team will do has the smallest error. Going to a
Cartesian GEI grid afterward is a single rigid SM->GEI rotation per frame
(exact), so the output frame is a free choice (--frame, default gei).

Each half-plane at azimuth phi is a grid in (rho >= 0, z), placed in SM at
  x_sm = rho cos(phi),  y_sm = rho sin(phi),  z_sm = z.
For r < 2.5 Re the foot point and |B| come from the analytic tilted dipole
(known directly from the SM coordinates); for r >= 2.5 Re they come from
BATSRUS's own theta_1/phi_1/Status and |B|, sampled once per frame for all
slices together. The RBE distribution at that foot point is Liouville /
first-invariant mapped to the local |B| (mapping.adiabatic_omni).

No NaNs: points outside the belt / on open lines / in the loss cone are a
true 0 (honest "no particles"); genuine interior pinholes (the ~0.2% off-grid
jitter) are filled from their mapped neighbours (binary_fill_holes + nearest).
"""

import os
import sys
import time
import math
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pvbatch's bundled python has no geopack; borrow the pure-python copy from the
# system site-packages (APPEND so pvbatch keeps its own numpy/scipy) -- the
# same shim footpoint_trace.py uses.
try:
    from geopack import geopack as _geopack_probe  # noqa: F401
except (ModuleNotFoundError, ValueError):
    import glob
    for _site_dir in sorted(glob.glob(
            os.path.expanduser('~/.local/lib/python*/site-packages'))):
        if not os.path.isdir(os.path.join(_site_dir, 'geopack')):
            continue
        sys.path.append(_site_dir)
        try:
            from geopack import geopack as _geopack_probe  # noqa: F401
            break
        except Exception:
            sys.path.remove(_site_dir)
            for _mod in [m for m in sys.modules if m.startswith('geopack')]:
                del sys.modules[_mod]

from swmf_vr import config, render, mapping, dipole, frames
from swmf_vr.slices_csv import (sm_to_gsm_matrix, sm_to_frame_matrix,
                                build_slice_points_sm, write_slices_csv, parse_bands)
import swmf_vr.data as data
import footpoint_volume as fpv
from slice3d import BANDS

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
EARTH_DIPOLE_TESLA = 3.03e-5           # B0 = bo * ro^3 measured in the RBE data
BAND_LABELS = [b[0] for b in BANDS]    # ['<1 MeV', '1-2 MeV', '>2 MeV']
# Plot color scale (>2 MeV omni), matching the cutaway colormap.json.
PLOT_VMIN, PLOT_VMAX = 1e3, 1e8


def dipole_foot_from_sm(pts_sm, handoff_re):
    """For SM points with 1 <= r < handoff: northern foot latitude (deg, +),
    MLT (hr) and local |B| (T) from the analytic tilted dipole -- read
    straight off the SM coordinates. NaN elsewhere."""
    n = len(pts_sm)
    foot_lat = np.full(n, np.nan); foot_mlt = np.full(n, np.nan)
    b_t = np.full(n, np.nan)
    r = np.linalg.norm(pts_sm, axis=1)
    sel = (r >= 1.0) & (r < handoff_re)
    for k in np.where(sel)[0]:
        x, y, zc = pts_sm[k]
        rk = r[k]
        lat = math.asin(zc / rk)                       # SM magnetic latitude
        phi_deg = math.degrees(math.atan2(y, x))
        l_shell = rk / math.cos(lat) ** 2
        foot_lat[k] = dipole.latitude_where_line_crosses_radius(l_shell, 1.0)
        foot_mlt[k] = (phi_deg / 15.0 + 12.0) % 24.0
        b_t[k] = EARTH_DIPOLE_TESLA / rk ** 3 * math.sqrt(1.0 + 3.0 * math.sin(lat) ** 2)
    return foot_lat, foot_mlt, b_t


def fill_pinholes(image):
    """image (n_z, n_rho): finite = mapped flux, NaN = unmapped. Interior
    holes (surrounded by mapped pixels) get their nearest mapped value;
    everything still unmapped becomes 0. Guarantees a finite array."""
    from scipy.ndimage import binary_fill_holes
    from scipy.interpolate import NearestNDInterpolator
    finite = np.isfinite(image)
    out = np.where(finite, image, 0.0)
    if not finite.any():
        return out
    interior = binary_fill_holes(finite)
    holes = interior & ~finite
    if holes.any():
        fz, fr = np.nonzero(finite)
        interp = NearestNDInterpolator(np.column_stack([fz, fr]), image[finite])
        hz, hr = np.nonzero(holes)
        out[hz, hr] = interp(np.column_stack([hz, hr]))
    return out


def map_points(pts_sm, R_sm2gsm, plt_path, rbe_grid, band_ch, widths, sin_grid,
               handoff_re, extent_re, quantity):
    """Flux (N, nbands), NaN where unmapped (open / off-grid / no flux)."""
    n = len(pts_sm)
    r = np.linalg.norm(pts_sm, axis=1)
    foot_lat = np.full(n, np.nan); foot_mlt = np.full(n, np.nan)
    b_tesla = np.full(n, np.nan); usable = np.zeros(n, dtype=bool)

    # r >= handoff: BATSRUS foot points, sampled at the GSM positions.
    outer = np.where((r >= handoff_re) & (r <= extent_re))[0]
    if len(outer):
        gsm = (R_sm2gsm @ pts_sm[outer].T).T
        arrays = fpv.sample_batsrus_at(plt_path, gsm)
        b_nt = np.sqrt(arrays['B_x_nT'] ** 2 + arrays['B_y_nT'] ** 2
                       + arrays['B_z_nT'] ** 2)
        foot_lat[outer] = arrays['theta_1_deg']
        foot_mlt[outer] = (arrays['phi_1_deg'] / 15.0 + 12.0) % 24.0
        b_tesla[outer] = b_nt * 1e-9
        ok = arrays['Status'] == fpv.CLOSED_STATUS
        if 'vtkValidPointMask' in arrays:
            ok &= arrays['vtkValidPointMask'] != 0
        usable[outer] = ok

    # 1 <= r < handoff: analytic dipole (direct from SM coords).
    d_lat, d_mlt, d_b = dipole_foot_from_sm(pts_sm, handoff_re)
    inner = np.isfinite(d_lat)
    foot_lat[inner] = d_lat[inner]; foot_mlt[inner] = d_mlt[inner]
    b_tesla[inner] = d_b[inner]; usable |= inner

    flux = np.full((n, len(BANDS)), np.nan)
    bratio = np.full(n, np.nan)       # local |B| / RBE B_min; < 1 = the two
    for k in np.where(usable)[0]:      # field-line mappings disagree (see doc)
        pad, bo = rbe_grid.at(foot_lat[k], foot_mlt[k])
        if pad is None:
            continue
        bratio[k] = b_tesla[k] / bo if bo and bo > 0 else np.nan
        for b, ch in enumerate(band_ch):
            if len(ch) == 0:
                continue
            j_eq = (pad[ch] * widths[ch, None]).sum(axis=0)
            if quantity == 'omni':
                val = mapping.adiabatic_omni(b_tesla[k], bo, j_eq, sin_grid)
            else:
                val = mapping.adiabatic_perp(b_tesla[k], bo, j_eq, sin_grid)
            if np.isfinite(val) and val > 0:
                flux[k, b] = val
    return flux, bratio


def preview_contact_sheet(out_png, azimuths_deg, rho, z, cube, band=2):
    """24-panel contact sheet of one frame's slices (>2 MeV omni by default)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    n = len(azimuths_deg)
    ncol = 6; nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow),
                             constrained_layout=True)
    norm = LogNorm(vmin=PLOT_VMIN, vmax=PLOT_VMAX)
    im = None
    for i, az in enumerate(azimuths_deg):
        ax = axes.ravel()[i]
        img = np.clip(cube[i, :, :, band], PLOT_VMIN, PLOT_VMAX)
        im = ax.pcolormesh(rho, z, img, norm=norm, cmap='inferno', shading='auto')
        ax.set_title('{}$^\\circ$'.format(int(az)), fontsize=9)
        ax.set_aspect('equal'); ax.set_xlim(rho[0], rho[-1])
        ax.set_ylim(z[0], z[-1])
        if i % ncol == 0:
            ax.set_ylabel('z (Re)')
        if i // ncol == nrow - 1:
            ax.set_xlabel(r'$\rho$ (Re)')
    for j in range(n, nrow * ncol):
        axes.ravel()[j].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                 label='{} omni flux [cm$^{{-2}}$ s$^{{-1}}$]'.format(BAND_LABELS[band]))
    fig.suptitle('Flux slices about the dipole axis', fontsize=13)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True, help='timestamp YYYYMMDD_HHMMSS (one frame)')
    p.add_argument('--step', type=float, default=0.1, help='in-plane grid spacing (Re)')
    p.add_argument('--extent', type=float, default=10.0, help='rho and |z| max (Re)')
    p.add_argument('--n-slices', type=int, default=24, help='azimuth cuts (360/N deg apart)')
    p.add_argument('--handoff', type=float, default=dipole.HANDOFF_RADIUS_RE)
    p.add_argument('--quantity', choices=['omni', 'perp'], default='omni')
    p.add_argument('--frame', choices=['gei', 'sm', 'gsm'], default='gei')
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--npz', default=None)
    p.add_argument('--plt-dir', default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--bands', default='all',
                   help='flux columns to write: all (default) or a comma list of lt1,1to2,gt2')
    p.add_argument('--drop-zeros', action='store_true',
                   help='omit CSV rows where every written flux column is 0')
    p.add_argument('--split-azimuths', action='store_true',
                   help='one CSV per cut (slice_azNNN_<ts>.csv) instead of one per timestamp')
    args = p.parse_args()
    bands = parse_bands(args.bands)

    cfg = config.CRAConfig()
    npz = os.path.abspath(args.npz) if args.npz else cfg.paths.npz_path
    plt_dir = os.path.abspath(args.plt_dir) if args.plt_dir else cfg.paths.plt_dir
    ts = args.ts

    csv_dir = os.path.join(args.out_root, 'fp_slices', 'csv')
    npz_dir = os.path.join(args.out_root, 'fp_slices', 'npz')
    for d in (csv_dir, npz_dir):
        os.makedirs(d, exist_ok=True)
    out_npz = os.path.join(npz_dir, 'slices_{}.npz'.format(ts))
    if not args.force and os.path.exists(out_npz):
        print('[{}] exists -- skip'.format(ts)); return

    plt_path = render.find_matching_plt(plt_dir, ts)
    if not plt_path or os.path.getsize(plt_path) == 0:
        print('[{}] no/empty .plt -- skip'.format(ts)); return

    base_dt = data.base_datetime(cfg.time.base_iso)
    frame_dt = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
    t = data.frame_index_for(npz, frame_dt, base_dt)   # npz time axis, any cadence

    zf = np.load(npz, allow_pickle=True)
    E_lvls = zf['E_lvls']
    sin_grid = np.asarray(zf['alpha_lvls'], dtype=float)
    n_frames = zf['bo'].shape[0]
    if t is None or not (0 <= t < n_frames):
        zf.close(); print('ts {} -> no flux frame ({})'.format(ts, t)); return
    lat_lvls = np.asarray(zf['lat_lvls'], dtype=float)
    mlt_lvls = np.asarray(zf['mlt_lvls'], dtype=float)
    bo_tesla = np.asarray(zf['bo'][t], dtype=float)
    irm_counts = np.asarray(zf['irm'][t, 0, :], dtype=float)
    zf.close()

    flux_frame = mapping.read_flux_frame(npz, t)
    if flux_frame is None:
        print('[{}] flux frame {} missing'.format(ts, t)); return
    rbe_grid = fpv.RbeFootpointGrid(lat_lvls, mlt_lvls, flux_frame, bo_tesla, irm_counts)
    band_ch = [np.where((E_lvls >= lo) & (E_lvls < hi))[0] for _, lo, hi in BANDS]
    widths = mapping.channel_widths(E_lvls)

    frames.recalc_for(frame_dt)             # set geopack dipole state for this frame
    R_sm2gsm = sm_to_gsm_matrix()
    R_out = sm_to_frame_matrix(args.frame)

    azimuths = np.arange(0.0, 360.0, 360.0 / args.n_slices)
    rho = np.arange(0.0, args.extent + args.step / 2, args.step)
    z = np.arange(-args.extent, args.extent + args.step / 2, args.step)
    pts_sm, n_rho, n_z = build_slice_points_sm(azimuths, rho, z)

    t0 = time.time()
    print('[{}] {} slices x {}x{} = {} points (step={} Re, dipole axis)'.format(
        ts, len(azimuths), n_z, n_rho, len(pts_sm), args.step))
    flux, bratio = map_points(pts_sm, R_sm2gsm, plt_path, rbe_grid, band_ch, widths,
                              sin_grid, args.handoff, args.extent, args.quantity)
    n_sub = int(np.sum(bratio[np.isfinite(bratio)] < 1.0))

    per_slice = n_rho * n_z
    cube = np.full((len(azimuths), n_z, n_rho, len(BANDS)), 0.0, dtype=np.float32)
    # |B|/B_min per pixel (0 where unmapped), saved so a "local B below RBE's
    # equatorial B" mask can be applied offline without recomputing.
    bratio_cube = np.zeros((len(azimuths), n_z, n_rho), dtype=np.float32)
    n_mapped = 0
    for si, az in enumerate(azimuths):
        block = flux[si * per_slice:(si + 1) * per_slice]      # (per_slice, nbands)
        block = block.reshape(n_z, n_rho, len(BANDS))
        br = bratio[si * per_slice:(si + 1) * per_slice].reshape(n_z, n_rho)
        bratio_cube[si] = np.where(np.isfinite(br), br, 0.0)
        for b in range(len(BANDS)):
            cube[si, :, :, b] = fill_pinholes(block[:, :, b])
        n_mapped += int(np.isfinite(block[:, :, 2]).sum())

    if args.split_azimuths:
        n_csv = len(azimuths)
        for si, az in enumerate(azimuths):
            out_csv = os.path.join(csv_dir, 'slice_az{:03d}_{}.csv'.format(int(round(az)), ts))
            write_slices_csv(out_csv, azimuths[si:si + 1], rho, z, cube[si:si + 1], R_out,
                             args.frame, ts, args.quantity, bands, args.drop_zeros)
    else:
        n_csv = 1
        out_csv = os.path.join(csv_dir, 'slices_{}.csv'.format(ts))
        write_slices_csv(out_csv, azimuths, rho, z, cube, R_out,
                         args.frame, ts, args.quantity, bands, args.drop_zeros)

    np.savez_compressed(out_npz, cube=cube, bratio=bratio_cube, azimuths_deg=azimuths.astype(np.float32),
             rho_re=rho.astype(np.float32), z_re=z.astype(np.float32),
             labels=BAND_LABELS, ts=ts, quantity=args.quantity,
             step=args.step, extent=args.extent, axis='SM_dipole', frame=args.frame)
    print('[{}] {} >2MeV mapped px ({} with |B| < B_min), {} csv file(s), {:.0f} s -> {}'.format(
        ts, n_mapped, n_sub, n_csv, time.time() - t0, out_npz))

    if not args.no_plot:
        out_png = os.path.join(args.out_root, 'fp_slices', 'preview_{}.png'.format(ts))
        preview_contact_sheet(out_png, azimuths, rho, z, cube)
        print('[{}] preview -> {}'.format(ts, out_png))


if __name__ == '__main__':
    main()
