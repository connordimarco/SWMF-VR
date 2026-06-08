#!/usr/bin/env python3
"""Stage A (system python3): equatorial flux slices + per-surface seed CSVs.

Reproduces v6/plot_flux_slices.py: for each timestep, extract the >=2 MeV flux
slice, find the inner/outer/danger boundaries, write the 2D flux plots, and write
the GSM seed CSVs that Stage B traces.

  source slurm/env.sh
  python3 scripts/stage_a_flux_seeds.py [rank size] [options]

Options (v6-compatible): --max-frames N --stride N --start N --force --no-zoom
--no-seeds, plus overrides --npz PATH --out-root DIR --energy-min KEV.
Sharding: streaming modulo (t_idx % size == rank).
"""

import os
import sys
import argparse
import datetime

# Make `import swmf_vr` work when run directly (PYTHONPATH also set by env.sh).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import config
from swmf_vr.io import npz as npzio
from swmf_vr import boundary, seeding, coords, plots
from swmf_vr.sharding import shard_modulo
from swmf_vr.timeaxis import base_datetime, frame_datetime


def build_config(args):
    cfg = config.default_config()
    paths = cfg.paths
    if args.npz:
        paths = paths._replace(npz_path=os.path.abspath(args.npz))
    if args.out_root:
        paths = paths._replace(out_root=os.path.abspath(args.out_root))
    cfg = cfg._replace(paths=paths)
    if args.energy_min is not None:
        cfg = cfg._replace(flux=cfg.flux._replace(energy_min_kev=args.energy_min))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shard_args', nargs='*', type=int)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-zoom', action='store_true')
    parser.add_argument('--no-seeds', action='store_true')
    parser.add_argument('--npz', default=None)
    parser.add_argument('--out-root', default=None)
    parser.add_argument('--energy-min', type=float, default=None)
    args = parser.parse_args()

    rank, size = 0, 1
    if len(args.shard_args) >= 2:
        rank, size = args.shard_args[0], args.shard_args[1]

    cfg = build_config(args)
    out_root = cfg.paths.out_root
    fig_dir = os.path.join(out_root, 'flux_slices')
    zoom_dir = os.path.join(out_root, 'flux_slices_zoom')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(zoom_dir, exist_ok=True)
    seed_dirs = {}
    for s in cfg.surfaces:
        d = os.path.join(out_root, s.seed_subdir)
        os.makedirs(d, exist_ok=True)
        seed_dirs[s.name] = d

    grids = npzio.load_grids(cfg.paths.npz_path)
    e_grid, a_grid, time_arr = grids['E_lvls'], grids['alpha_lvls'], grids['time']
    ro_vals, mlto_vals = grids['ro'], grids['mlto']
    e_idxs, a_idx = npzio.select_energy_alpha(
        e_grid, a_grid, cfg.flux.energy_min_kev, cfg.flux.target_alpha_val)
    base_dt = base_datetime(cfg.time.base_iso)

    print('Energy channels >= {:.0f} keV: {}'.format(
        cfg.flux.energy_min_kev, e_grid[e_idxs].tolist()))
    print('Worker {}/{}'.format(rank, size))

    rendered = 0
    for t_idx, frame in npzio.iter_flux_frames(cfg.paths.npz_path):
        if t_idx >= len(time_arr):
            break
        if t_idx < args.start:
            continue
        if not shard_modulo(t_idx, rank, size):
            continue
        if args.stride > 1 and ((t_idx - args.start) // size) % args.stride != 0:
            continue

        cur_dt = frame_datetime(time_arr[t_idx], base_dt)
        ts_str = cur_dt.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(fig_dir, 'flux_slice_{}.png'.format(ts_str))
        zoom_out_path = os.path.join(zoom_dir, 'flux_slice_zoom_{}.png'.format(ts_str))

        done = os.path.exists(out_path)
        if not args.no_zoom:
            done = done and os.path.exists(zoom_out_path)
        if not args.no_seeds:
            for s in cfg.surfaces:
                done = done and os.path.exists(
                    os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts_str)))
        if done and not args.force:
            rendered += 1
            continue

        flux_slice = npzio.extract_integrated_flux(
            frame, e_idxs, a_idx, cfg.flux.lat_dim, cfg.flux.mlt_dim)

        boundary_lines = {}
        for s in cfg.surfaces:
            if s.mode == 'max':
                boundary_lines[(s.name, 'max')] = boundary.find_max_boundary(
                    flux_slice, ro_vals)
            else:
                for threshold in s.thresholds:
                    boundary_lines[(s.name, threshold)] = boundary.find_boundary(
                        flux_slice, ro_vals, threshold, s.search)

        plots.plot_frame(ro_vals, mlto_vals, flux_slice, boundary_lines,
                         cfg.surfaces, ts_str, out_path)
        if not args.no_zoom:
            plots.plot_frame(ro_vals, mlto_vals, flux_slice, boundary_lines,
                             cfg.surfaces, ts_str, zoom_out_path,
                             radial_limits=(1.0, 4.0), title_suffix='zoom 1.0-4.0 Re')

        if not args.no_seeds:
            coords.recalc_for(cur_dt)   # geopack.recalc before SM->GSM
            mlt_centers = mlto_vals[0, :]
            peak_arr = boundary.find_peak_flux(flux_slice, ro_vals)
            for s in cfg.surfaces:
                if s.mode == 'max':
                    r_arr = boundary_lines[(s.name, 'max')]
                    flux_label = peak_arr
                else:
                    label = s.thresholds[0]
                    r_arr = boundary_lines[(s.name, label)]
                    flux_label = float(label)
                seed_path = os.path.join(seed_dirs[s.name],
                                         'boundary_{}.csv'.format(ts_str))
                seeding.write_seed_csv(r_arr, mlt_centers, flux_label, seed_path,
                                       flux_peak=peak_arr)

        rendered += 1
        if rendered % 25 == 0:
            print('Worker {}: rendered {} frames'.format(rank, rendered))
        if args.max_frames is not None and rendered >= args.max_frames:
            break

    print('Worker {}: done, rendered {} frames'.format(rank, rendered))


if __name__ == '__main__':
    main()
