#!/usr/bin/env python3
"""CIMI equatorial flux -> 3D, sampled on the y=0 GSM plane (pvbatch).

For each point in the noon-midnight meridian, trace the BATSRUS field line and
map CIMI's equatorial flux out to it -- B_s locally, B_eq = min|B| along the line,
the equatorial footpoint -> CIMI PAD lookup -> omnidirectional (or perpendicular)
flux, integrated into <1 / 1-2 / >2 MeV bands. Open lines are masked. Each frame
writes a 3-panel PNG (png/) and an .npz of the raw slice (npz/, so the despeckle +
plot can be re-tuned offline via replot_slice.py without re-tracing).

  "$PV_BATCH" scripts/slice3d.py [rank size]   # full run, frames sharded rank::size
  "$PV_BATCH" scripts/slice3d.py --ts 20240510_210000   # just one frame
       [--n N] [--extent RE] [--quantity omni|perp] [--out-root DIR] [--force]
"""

import os
import sys
import datetime
import argparse
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import config, render, mapping, data, sliceplot

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except Exception:                                  # older bundling
    from vtk.util.numpy_support import vtk_to_numpy

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BANDS = [('<1 MeV', 0.0, 1000.0), ('1-2 MeV', 1000.0, 2000.0), ('>2 MeV', 2000.0, 1e12)]


def plane_seeds(extent, n, r_min=2.05):
    """Grid of (x,0,z) points in the y=0 plane, skipping inside the inner body.
    Returns (seeds (K,3), ij (K,2) -> (iz,ix), xs, zs)."""
    xs = np.linspace(-extent, extent, n)
    zs = np.linspace(-extent, extent, n)
    seeds, ij = [], []
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            if x * x + z * z >= r_min * r_min:
                seeds.append((float(x), 0.0, float(z)))
                ij.append((iz, ix))
    return np.array(seeds), np.array(ij), xs, zs


def per_seed_fields(poly, seeds, foot_rmax=8.0):
    """From the traced streamlines, per seed: B_eq (min|B| within the belt, i.e.
    r <= foot_rmax, so a tail/cusp excursion can't masquerade as the equator),
    its footpoint xyz, max r along the whole line (open/closed test), and B_s
    (|B| nearest the seed)."""
    ns = len(seeds)
    b_eq = np.full(ns, np.inf)
    foot = np.zeros((ns, 3))
    maxr = np.zeros(ns)
    bs_d = np.full(ns, np.inf)
    b_s = np.full(ns, np.nan)

    pts = vtk_to_numpy(poly.GetPoints().GetData())
    Barr = poly.GetPointData().GetArray('B')
    if Barr is not None:
        bmag_all = np.linalg.norm(vtk_to_numpy(Barr), axis=1)
    else:                                          # fall back to components
        comp = [vtk_to_numpy(poly.GetPointData().GetArray(c))
                for c in ('B_x_nT', 'B_y_nT', 'B_z_nT')]
        bmag_all = np.sqrt(sum(c * c for c in comp))
    r_all = np.linalg.norm(pts, axis=1)
    seedids = vtk_to_numpy(poly.GetCellData().GetArray('SeedIds'))

    for c in range(poly.GetNumberOfCells()):
        ids = poly.GetCell(c).GetPointIds()
        idx = [ids.GetId(k) for k in range(ids.GetNumberOfIds())]
        if not idx:
            continue
        sid = int(seedids[c])
        PP, bb, rr = pts[idx], bmag_all[idx], r_all[idx]
        belt = np.where(rr <= foot_rmax)[0]            # equator must be in the belt
        if belt.size:
            jmin = belt[int(bb[belt].argmin())]
            if bb[jmin] < b_eq[sid]:
                b_eq[sid], foot[sid] = bb[jmin], PP[jmin]
        mr = float(rr.max())
        if mr > maxr[sid]:
            maxr[sid] = mr
        d = ((PP - seeds[sid]) ** 2).sum(axis=1)
        dmin = int(d.argmin())
        if d[dmin] < bs_d[sid]:
            bs_d[sid], b_s[sid] = d[dmin], bb[dmin]
    return b_eq, foot, maxr, b_s


def map_seeds(b_eq, foot, maxr, b_s, cimi, ctx, args):
    """Adiabatically map every closed seed into the energy bands. Returns the per-
    band (n,n) flux arrays plus (n_closed, n_mapped, B_eq/bo ratios)."""
    flux = [np.full((args.n, args.n), np.nan) for _ in BANDS]
    n_closed = n_mapped = 0
    ratios = []
    for sid in range(len(b_eq)):
        if not np.isfinite(b_eq[sid]) or maxr[sid] > args.open_rmax:
            continue                                # no trace, or open line
        n_closed += 1
        fx, fy, fz = foot[sid]
        r_eq = float(np.sqrt(fx * fx + fy * fy + fz * fz))
        mlt_eq = float((np.degrees(np.arctan2(fy, fx)) / 15.0 + 12.0) % 24.0)
        pad, bo_c = cimi.at(r_eq, mlt_eq)           # equatorial PAD (E, A), B_eq
        if pad is None:
            continue
        iz, ix = ctx.ij[sid]
        mapped_any = False
        for b, ch in enumerate(ctx.band_ch):
            if len(ch) == 0:
                continue
            j_eq = (pad[ch] * ctx.widths[ch, None]).sum(axis=0)   # integrate band -> (A,)
            if args.quantity == 'omni':
                val = mapping.adiabatic_omni(b_s[sid], b_eq[sid], j_eq, ctx.sin_grid)
            else:
                val = mapping.adiabatic_perp(b_s[sid], b_eq[sid], j_eq, ctx.sin_grid)
            if np.isfinite(val) and val > 0:
                flux[b][iz, ix] = val
                mapped_any = True
        if mapped_any:
            n_mapped += 1
        if bo_c and bo_c > 0:
            ratios.append(b_eq[sid] * 1e-9 / bo_c)  # trace |B| nT->T vs CIMI bo (T)
    return flux, n_closed, n_mapped, ratios


def process_frame(ts, flux_frame, ro, mlto, bo, ctx, args):
    """Trace + map + save one frame. Returns True on success."""
    plt_path = render.find_matching_plt(ctx.plt_dir, ts)
    if not plt_path:
        print('[{}] no .plt -- skip'.format(ts)); return False
    cimi = mapping.CimiEquator(ro, mlto, flux_frame, bo=bo, dmax=args.max_dist)

    seed_csv = os.path.join(ctx.seeds_dir, '_seeds_{}.csv'.format(ts))   # per-frame, collision-free
    with open(seed_csv, 'w') as f:
        f.write('Seed_X_Re,Seed_Y_Re,Seed_Z_Re\n')
        for x, y, z in ctx.seeds:
            f.write('{:.5f},{:.5f},{:.5f}\n'.format(x, y, z))

    reader, b_calc = render.load_b_field(plt_path)
    trace_cfg = ctx.cfg.trace._replace(max_streamline_length=args.max_len)
    res = render.trace_streamlines(b_calc, seed_csv, trace_cfg)
    if res is None:
        for pr in (b_calc, reader):
            render.Delete(pr)
        os.remove(seed_csv)
        print('[{}] trace failed'.format(ts)); return False
    tracer, points_source, csv_reader, poly, _ = res
    b_eq, foot, maxr, b_s = per_seed_fields(poly, ctx.seeds, foot_rmax=args.foot_rmax)
    flux, n_closed, n_mapped, ratios = map_seeds(b_eq, foot, maxr, b_s, cimi, ctx, args)
    for pr in (tracer, points_source, csv_reader, b_calc, reader):
        render.Delete(pr)
    os.remove(seed_csv)

    np.savez(os.path.join(ctx.npz_dir, 'slice_y0_{}_bands.npz'.format(ts)),
             xs=ctx.xs, zs=ctx.zs, flux=np.array(flux), labels=ctx.labels,
             ts=ts, quantity=args.quantity)
    png = os.path.join(ctx.png_dir, 'slice_y0_{}_bands_{}.png'.format(ts, args.quantity))
    sliceplot.plot_bands(ctx.xs, ctx.zs, flux, ctx.labels, ts, args.quantity, png,
                         indices=ctx.indices)
    rat = '  |B|eq/bo med {:.2f}'.format(float(np.median(ratios))) if ratios else ''
    print('[{}] mapped {}/{}{}'.format(ts, n_mapped, n_closed, rat))
    return True


def main():
    # ONE frame per process. ParaView accumulates memory across pipeline rebuilds,
    # so processing a whole shard in one long-lived process OOMs after a few frames
    # -- the array script (slurm/slice_array.sh) loops and invokes this once per
    # frame instead, so memory is reclaimed every frame.
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True, help='timestamp YYYYMMDD_HHMMSS (one frame)')
    p.add_argument('--n', type=int, default=121)
    p.add_argument('--extent', type=float, default=10.0)
    p.add_argument('--quantity', choices=['omni', 'perp'], default='omni')
    p.add_argument('--open-rmax', type=float, default=12.0)
    p.add_argument('--foot-rmax', type=float, default=8.0)
    p.add_argument('--max-len', type=float, default=50.0)
    p.add_argument('--max-dist', type=float, default=1.0)
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--npz', default=None)
    p.add_argument('--plt-dir', default=None)
    args = p.parse_args()

    cfg = config.CRAConfig()
    npz = os.path.abspath(args.npz) if args.npz else cfg.paths.npz_path
    plt_dir = os.path.abspath(args.plt_dir) if args.plt_dir else cfg.paths.plt_dir

    # frame index == minutes after base (1-min cadence; the NPZ 'time' array is an
    # unreadable object pickle, but the .plt names confirm the cadence).
    base_dt = data.base_datetime(cfg.time.base_iso)
    t = int(round((datetime.datetime.strptime(args.ts, '%Y%m%d_%H%M%S')
                   - base_dt).total_seconds() / 60.0))
    zf = np.load(npz, allow_pickle=True)
    E_lvls = zf['E_lvls']
    sin_grid = np.asarray(zf['alpha_lvls'], dtype=float)
    n_frames = zf['ro'].shape[0]
    if not (0 <= t < n_frames):
        zf.close(); print('ts {} -> frame {} out of range'.format(args.ts, t)); return
    ro, mlto, bo = zf['ro'][t], zf['mlto'][t], zf['bo'][t]
    zf.close()

    band_ch = [np.where((E_lvls >= lo) & (E_lvls < hi))[0] for _, lo, hi in BANDS]
    seeds, ij, xs, zs = plane_seeds(args.extent, args.n)
    png_dir = os.path.join(args.out_root, 'slice_renders', 'png')
    npz_dir = os.path.join(args.out_root, 'slice_renders', 'npz')
    seeds_dir = os.path.join(args.out_root, 'slice_renders', 'seeds')
    for d in (png_dir, npz_dir, seeds_dir):
        os.makedirs(d, exist_ok=True)

    indices_npz = os.path.join(args.out_root, 'slice_renders', 'indices.npz')
    ctx = SimpleNamespace(cfg=cfg, plt_dir=plt_dir, seeds=seeds, ij=ij, xs=xs, zs=zs,
                          band_ch=band_ch, widths=mapping.channel_widths(E_lvls),
                          sin_grid=sin_grid, labels=[b[0] for b in BANDS],
                          seeds_dir=seeds_dir, png_dir=png_dir, npz_dir=npz_dir,
                          indices=indices_npz if os.path.exists(indices_npz) else None)

    process_frame(args.ts, mapping.read_flux_frame(npz, t), ro, mlto, bo, ctx, args)


if __name__ == '__main__':
    main()
