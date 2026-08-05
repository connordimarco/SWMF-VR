#!/usr/bin/env python3
"""CIMI equatorial flux -> 3D voxel grid (pvbatch), the format the VR team asked for.

Same trace + adiabatic mapping as slice3d.py, but the samples are a full 3D
lattice (GSM Re, Earth-centered, clipped to r_min..extent) instead of the y=0
plane. Per frame writes:

  csv/volume_<ts>_bands.csv   one line per point: x,y,z then the three band
                              fluxes (<1 / 1-2 / >2 MeV). NaN = no value there
                              (open line, inside the inner body, or footpoint
                              off the CIMI grid).
  npz/volume_<ts>_bands.npz   the raw points+flux, so the CSV can be
                              reformatted / rotated to GEI offline without
                              re-tracing.

  "$PV_BATCH" scripts/volume3d.py --ts 20240510_210000
       [--step RE] [--extent RE] [--quantity omni|perp] [--out-root DIR] [--force]
"""

import os
import sys
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swmf_vr import config, render, mapping
from slice3d import BANDS, per_seed_fields

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def volume_seeds(extent, step, r_min=2.05):
    """3D lattice of (x,y,z) GSM points with r_min <= r <= extent (the corners
    beyond `extent` are open-line territory anyway; the inner body can't trace)."""
    ax = np.linspace(-extent, extent, int(round(2 * extent / step)) + 1)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    r = np.linalg.norm(pts, axis=1)
    return pts[(r >= r_min) & (r <= extent)]


def map_volume(b_eq, foot, maxr, b_s, cimi, band_ch, widths, sin_grid, args):
    """Adiabatically map every closed seed. Returns ((K, nbands) flux, NaN where
    unmapped, plus (n_closed, n_mapped, B_eq/bo ratios))."""
    K = len(b_eq)
    flux = np.full((K, len(BANDS)), np.nan)
    n_closed = n_mapped = 0
    ratios = []
    for sid in range(K):
        if not np.isfinite(b_eq[sid]) or maxr[sid] > args.open_rmax:
            continue                                # no trace, or open line
        n_closed += 1
        fx, fy, fz = foot[sid]
        r_eq = float(np.sqrt(fx * fx + fy * fy + fz * fz))
        mlt_eq = float((np.degrees(np.arctan2(fy, fx)) / 15.0 + 12.0) % 24.0)
        pad, bo_c = cimi.at(r_eq, mlt_eq)           # equatorial PAD (E, A), B_eq
        if pad is None:
            continue
        mapped_any = False
        for b, ch in enumerate(band_ch):
            if len(ch) == 0:
                continue
            j_eq = (pad[ch] * widths[ch, None]).sum(axis=0)   # integrate band -> (A,)
            if args.quantity == 'omni':
                val = mapping.adiabatic_omni(b_s[sid], b_eq[sid], j_eq, sin_grid)
            else:
                val = mapping.adiabatic_perp(b_s[sid], b_eq[sid], j_eq, sin_grid)
            if np.isfinite(val) and val > 0:
                flux[sid, b] = val
                mapped_any = True
        if mapped_any:
            n_mapped += 1
        if bo_c and bo_c > 0:
            ratios.append(b_eq[sid] * 1e-9 / bo_c)  # trace |B| nT->T vs CIMI bo (T)
    return flux, n_closed, n_mapped, ratios


def write_csv(path, seeds, flux, ts, args):
    unit = 'particles/(cm^2 s)' if args.quantity == 'omni' else 'particles/(cm^2 s sr)'
    def fmt(v):
        return 'NaN' if not np.isfinite(v) else '{:.5g}'.format(v)
    with open(path, 'w') as f:
        f.write('# SWMF-VR 3D flux volume  ts={}  frame=GSM (Re, Earth-centered)\n'.format(ts))
        f.write('# grid: step={} Re, {} <= r <= {} Re;  {} energy-band-integrated '
                'flux, {}\n'.format(args.step, 2.05, args.extent, args.quantity, unit))
        f.write('# NaN = no value (open field line or equatorial footpoint off the CIMI grid)\n')
        f.write('x_re,y_re,z_re,flux_lt1MeV,flux_1to2MeV,flux_gt2MeV\n')
        for (x, y, z), row in zip(seeds, flux):
            f.write('{:.3f},{:.3f},{:.3f},{},{},{}\n'.format(
                x, y, z, fmt(row[0]), fmt(row[1]), fmt(row[2])))


def main():
    # ONE frame per process (same reason as slice3d: ParaView leaks across
    # pipeline rebuilds; per-frame processes reclaim it).
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True, help='timestamp YYYYMMDD_HHMMSS (one frame)')
    p.add_argument('--step', type=float, default=0.5, help='grid spacing (Re)')
    p.add_argument('--extent', type=float, default=10.0)
    p.add_argument('--quantity', choices=['omni', 'perp'], default='omni')
    p.add_argument('--open-rmax', type=float, default=12.0)
    p.add_argument('--foot-rmax', type=float, default=8.0)
    p.add_argument('--max-len', type=float, default=50.0)
    p.add_argument('--max-dist', type=float, default=1.0)
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--npz', default=None)
    p.add_argument('--plt-dir', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    cfg = config.CRAConfig()
    npz = os.path.abspath(args.npz) if args.npz else cfg.paths.npz_path
    plt_dir = os.path.abspath(args.plt_dir) if args.plt_dir else cfg.paths.plt_dir

    csv_dir = os.path.join(args.out_root, 'volume_renders', 'csv')
    npz_dir = os.path.join(args.out_root, 'volume_renders', 'npz')
    seeds_dir = os.path.join(args.out_root, 'volume_renders', 'seeds')
    for d in (csv_dir, npz_dir, seeds_dir):
        os.makedirs(d, exist_ok=True)
    ts = args.ts
    out_csv = os.path.join(csv_dir, 'volume_{}_bands.csv'.format(ts))
    out_npz = os.path.join(npz_dir, 'volume_{}_bands.npz'.format(ts))
    if not args.force and os.path.exists(out_csv) and os.path.exists(out_npz):
        print('[{}] exists -- skip'.format(ts)); return

    plt_path = render.find_matching_plt(plt_dir, ts)
    if not plt_path:
        print('[{}] no .plt -- skip'.format(ts)); return

    # frame index == minutes after base (1-min cadence, same as slice3d)
    import swmf_vr.data as data
    base_dt = data.base_datetime(cfg.time.base_iso)
    t = int(round((datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
                   - base_dt).total_seconds() / 60.0))
    zf = np.load(npz, allow_pickle=True)
    E_lvls = zf['E_lvls']
    sin_grid = np.asarray(zf['alpha_lvls'], dtype=float)
    n_frames = zf['ro'].shape[0]
    if not (0 <= t < n_frames):
        zf.close(); print('ts {} -> frame {} out of range'.format(ts, t)); return
    ro, mlto, bo = zf['ro'][t], zf['mlto'][t], zf['bo'][t]
    zf.close()

    band_ch = [np.where((E_lvls >= lo) & (E_lvls < hi))[0] for _, lo, hi in BANDS]
    widths = mapping.channel_widths(E_lvls)
    seeds = volume_seeds(args.extent, args.step)
    print('[{}] tracing {} volume points (step={} Re)'.format(ts, len(seeds), args.step))

    cimi = mapping.CimiEquator(ro, mlto, mapping.read_flux_frame(npz, t),
                               bo=bo, dmax=args.max_dist)

    seed_csv = os.path.join(seeds_dir, '_seeds_{}.csv'.format(ts))
    with open(seed_csv, 'w') as f:
        f.write('Seed_X_Re,Seed_Y_Re,Seed_Z_Re\n')
        for x, y, z in seeds:
            f.write('{:.5f},{:.5f},{:.5f}\n'.format(x, y, z))

    reader, b_calc = render.load_b_field(plt_path)
    trace_cfg = cfg.trace._replace(max_streamline_length=args.max_len)
    res = render.trace_streamlines(b_calc, seed_csv, trace_cfg)
    if res is None:
        for pr in (b_calc, reader):
            render.Delete(pr)
        os.remove(seed_csv)
        print('[{}] trace failed'.format(ts)); return
    tracer, points_source, csv_reader, poly, _ = res
    b_eq, foot, maxr, b_s = per_seed_fields(poly, seeds, foot_rmax=args.foot_rmax)
    flux, n_closed, n_mapped, ratios = map_volume(
        b_eq, foot, maxr, b_s, cimi, band_ch, widths, sin_grid, args)
    for pr in (tracer, points_source, csv_reader, b_calc, reader):
        render.Delete(pr)
    os.remove(seed_csv)

    write_csv(out_csv, seeds, flux, ts, args)
    np.savez(out_npz, points=seeds.astype(np.float32), flux=flux.astype(np.float32),
             labels=[b[0] for b in BANDS], ts=ts, quantity=args.quantity,
             step=args.step, extent=args.extent, frame='GSM')
    rat = '  |B|eq/bo med {:.2f}'.format(float(np.median(ratios))) if ratios else ''
    print('[{}] mapped {}/{} closed of {} points{}  -> {}'.format(
        ts, n_mapped, n_closed, len(seeds), rat, out_csv))


if __name__ == '__main__':
    main()
