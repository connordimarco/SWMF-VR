#!/usr/bin/env python3
"""Seed CSVs -> traced, shaded field-line OBJs + a colored screenshot (pvbatch).

  "$PV_BATCH" scripts/trace.py [rank size [max_files]] [--match SUBSTR]
       [--color-by NAME] [--cmap NAME] [--vmin V] [--vmax V] [--log/--no-log]
       [--out-root DIR] [--plt-dir DIR]

rank/size shard by slicing the sorted timestamp list (stamps[rank::size]).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from paraview.simple import Show, Render, SaveScreenshot, Delete

from swmf_vr import config, render
from swmf_vr.color import write_colored_obj, write_color_assets
from swmf_vr.data import is_header_only


def parse_args(argv):
    rank, size, max_files, match = 0, 1, None, None
    over, out_root, plt_dir = {}, None, None
    ints, i = [], 0
    while i < len(argv) and len(ints) < 3:
        try:
            ints.append(int(argv[i])); i += 1
        except ValueError:
            break
    if len(ints) >= 2:
        rank, size = ints[0], ints[1]
    if len(ints) >= 3:
        max_files = ints[2]
    while i < len(argv):
        a = argv[i]
        if a == '--match': match = argv[i + 1]; i += 2; continue
        if a == '--color-by': over['scalar'] = argv[i + 1]; i += 2; continue
        if a == '--cmap': over['cmap'] = argv[i + 1]; i += 2; continue
        if a == '--vmin': over['vmin'] = float(argv[i + 1]); i += 2; continue
        if a == '--vmax': over['vmax'] = float(argv[i + 1]); i += 2; continue
        if a == '--log': over['log'] = True; i += 1; continue
        if a == '--no-log': over['log'] = False; i += 1; continue
        if a == '--out-root': out_root = argv[i + 1]; i += 2; continue
        if a == '--plt-dir': plt_dir = argv[i + 1]; i += 2; continue
        i += 1
    return rank, size, max_files, match, over, out_root, plt_dir


def main():
    rank, size, max_files, match, over, out_root, plt_dir = parse_args(sys.argv[1:])
    cfg = config.CRAConfig()
    if over:
        cfg = cfg._replace(shading=cfg.shading._replace(**over))
    if out_root:
        cfg = cfg._replace(paths=cfg.paths._replace(out_root=os.path.abspath(out_root)))
    if plt_dir:
        cfg = cfg._replace(paths=cfg.paths._replace(plt_dir=os.path.abspath(plt_dir)))

    scalar = cfg.shading.scalar
    if scalar not in (set(config.SEED_SCALARS) | set(config.POINT_SCALARS)):
        print("unknown --color-by '{}'".format(scalar)); sys.exit(1)
    log, vmin, vmax = config.resolve_color_range(cfg.shading)
    print('color: {} {} [{:g},{:g}] {}'.format(
        scalar, cfg.shading.cmap, vmin, vmax, 'log' if log else 'linear'))

    out = cfg.paths.out_root
    seed_dirs = {s.name: os.path.join(out, s.seed_subdir) for s in cfg.surfaces}
    obj_dirs = {s.name: os.path.join(out, s.obj_subdir) for s in cfg.surfaces}
    for d in obj_dirs.values():
        os.makedirs(d, exist_ok=True)
    shot_dir = os.path.join(out, 'screenshot_renders')
    os.makedirs(shot_dir, exist_ok=True)
    view = render.make_view(cfg.camera)

    stamps = render.collect_timestamps(seed_dirs.values())
    if match is not None:
        stamps = [s for s in stamps if match in s]
    if not stamps:
        print('no seed CSVs; run seeds.py first'); return
    mine = stamps[rank::size]
    if max_files is not None:
        mine = mine[:max_files]
    print('{} stamps total; worker {}/{} -> {}'.format(len(stamps), rank, size, len(mine)))

    for ts in mine:
        one(cfg, ts, seed_dirs, obj_dirs, shot_dir, view, scalar, log, vmin, vmax)
    print('worker {}: done'.format(rank))


def one(cfg, ts, seed_dirs, obj_dirs, shot_dir, view, scalar, log, vmin, vmax):
    plt_path = render.find_matching_plt(cfg.paths.plt_dir, ts)
    if not plt_path:
        print('[{}] no .plt -- skipping'.format(ts)); return
    reader = b_calc = None
    proxies, shown = [], False
    try:
        reader, b_calc = render.load_b_field(plt_path)
        for s in cfg.surfaces:
            csv_path = os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts))
            if not os.path.exists(csv_path) or is_header_only(csv_path):
                continue
            res = render.trace_streamlines(b_calc, csv_path, cfg.trace)
            if res is None:
                continue
            tracer, points_source, csv_reader, poly, seed_poly = res
            proxies += [tracer, points_source, csv_reader]
            points, cells, vals = render.extract_geometry_and_scalar(poly, seed_poly, scalar)
            out_path = os.path.join(obj_dirs[s.name], 'fieldlines_{}.obj'.format(ts))
            umin, umax = write_colored_obj(points, cells, vals, out_path, scalar, log, vmin, vmax)
            write_color_assets(obj_dirs[s.name], cfg.shading.cmap, scalar, log, vmin, vmax)
            print('[{}] {}: {} cells (U {:.3f}-{:.3f})'.format(ts, s.name, len(cells), umin, umax))
            pv = render.show_scalar_colored(view, poly, vals, cfg.shading)
            if pv is not None:
                proxies.append(pv)
            else:
                disp = Show(tracer, view); disp.DiffuseColor = list(s.rgb); disp.AmbientColor = list(s.rgb)
            shown = True
        if shown:
            Render()
            SaveScreenshot(os.path.join(shot_dir, 'fieldlines_{}.png'.format(ts)), view)
    except Exception as e:
        print('[{}] failed ({})'.format(ts, e))
    finally:
        for proxy in proxies:
            Delete(proxy)
        if b_calc is not None:
            Delete(b_calc)
        if reader is not None:
            Delete(reader)


if __name__ == '__main__':
    main()
