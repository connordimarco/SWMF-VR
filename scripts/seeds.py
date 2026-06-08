#!/usr/bin/env python3
"""Raw flux -> boundary seed CSVs + 2D flux plots (system python3).

  python3 scripts/seeds.py [rank size] [--start N] [--max-frames N] [--stride N]
                           [--force] [--no-zoom] [--no-seeds]
                           [--npz PATH] [--out-root DIR] [--energy-min KEV]

rank/size shard by frame: a worker takes frames where idx % size == rank.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import config, belt, frames, color
from swmf_vr import data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('shard', nargs='*', type=int)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-zoom', action='store_true')
    p.add_argument('--no-seeds', action='store_true')
    p.add_argument('--npz', default=None)
    p.add_argument('--out-root', default=None)
    p.add_argument('--energy-min', type=float, default=None)
    args = p.parse_args()

    rank, size = (args.shard + [0, 1])[:2]

    cfg = config.CRAConfig()
    if args.npz:
        cfg = cfg._replace(paths=cfg.paths._replace(npz_path=os.path.abspath(args.npz)))
    if args.out_root:
        cfg = cfg._replace(paths=cfg.paths._replace(out_root=os.path.abspath(args.out_root)))
    if args.energy_min is not None:
        cfg = cfg._replace(flux=cfg.flux._replace(energy_min_kev=args.energy_min))

    out = cfg.paths.out_root
    fig_dir = os.path.join(out, 'flux_slices')
    zoom_dir = os.path.join(out, 'flux_slices_zoom')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(zoom_dir, exist_ok=True)
    seed_dirs = {}
    for s in cfg.surfaces:
        d = os.path.join(out, s.seed_subdir)
        os.makedirs(d, exist_ok=True)
        seed_dirs[s.name] = d

    g = data.load_grids(cfg.paths.npz_path)
    time_arr, ro_vals, mlto_vals = g['time'], g['ro'], g['mlto']
    e_idxs, a_idx = data.select_energy_alpha(
        g['E_lvls'], g['alpha_lvls'], cfg.flux.energy_min_kev, cfg.flux.target_alpha_val)
    base_dt = data.base_datetime(cfg.time.base_iso)
    print('energy channels >= {:.0f} keV: {}'.format(
        cfg.flux.energy_min_kev, g['E_lvls'][e_idxs].tolist()))
    print('worker {}/{}'.format(rank, size))

    done = 0
    for t_idx, frame in data.iter_flux_frames(cfg.paths.npz_path):
        if t_idx >= len(time_arr):
            break
        if t_idx < args.start or t_idx % size != rank:
            continue
        if args.stride > 1 and ((t_idx - args.start) // size) % args.stride != 0:
            continue

        ts = data.frame_datetime(time_arr[t_idx], base_dt).strftime('%Y%m%d_%H%M%S')
        out_png = os.path.join(fig_dir, 'flux_slice_{}.png'.format(ts))
        zoom_png = os.path.join(zoom_dir, 'flux_slice_zoom_{}.png'.format(ts))

        have = os.path.exists(out_png)
        if not args.no_zoom:
            have = have and os.path.exists(zoom_png)
        if not args.no_seeds:
            for s in cfg.surfaces:
                have = have and os.path.exists(
                    os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts)))
        if have and not args.force:
            done += 1
            continue

        flux_slice = data.extract_integrated_flux(
            frame, e_idxs, a_idx, cfg.flux.lat_dim, cfg.flux.mlt_dim)
        boundary_lines = {}
        for s in cfg.surfaces:
            if s.mode == 'max':
                boundary_lines[(s.name, 'max')] = belt.find_max_boundary(flux_slice, ro_vals)
            else:
                for th in s.thresholds:
                    boundary_lines[(s.name, th)] = belt.find_boundary(flux_slice, ro_vals, th, s.search)

        color.plot_frame(ro_vals, mlto_vals, flux_slice, boundary_lines, cfg.surfaces, ts, out_png)
        if not args.no_zoom:
            color.plot_frame(ro_vals, mlto_vals, flux_slice, boundary_lines, cfg.surfaces,
                             ts, zoom_png, radial_limits=(1.0, 4.0), title_suffix='zoom 1.0-4.0 Re')

        if not args.no_seeds:
            frames.recalc_for(data.frame_datetime(time_arr[t_idx], base_dt))
            mlt_centers = mlto_vals[0, :]
            peak = belt.find_peak_flux(flux_slice, ro_vals)
            for s in cfg.surfaces:
                if s.mode == 'max':
                    r_arr, flux_label = boundary_lines[(s.name, 'max')], peak
                else:
                    r_arr, flux_label = boundary_lines[(s.name, s.thresholds[0])], float(s.thresholds[0])
                belt.write_seed_csv(r_arr, mlt_centers, flux_label,
                                    os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts)),
                                    flux_peak=peak)

        done += 1
        if args.max_frames is not None and done >= args.max_frames:
            break

    print('worker {}: {} frames'.format(rank, done))


if __name__ == '__main__':
    main()
