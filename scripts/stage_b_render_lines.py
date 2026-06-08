#!/usr/bin/env python3
"""Stage B (ParaView pvbatch): trace field-line shells and write colored OBJ.

Reproduces v6/render_field_lines_v6.py: for each timestep with seeds, trace the
inner/outer/danger surfaces through the GM B-field, bake the chosen color scalar
into OBJ `vt` (+ ramp assets), and save a scalar-colored screenshot.

  source slurm/env.sh
  "$PV_BATCH" scripts/stage_b_render_lines.py [rank size [max_files]] [options]

Options (v6-compatible): --match SUBSTR --color-by NAME --cmap NAME --vmin V
--vmax V --log/--no-log, plus --out-root DIR --plt-dir DIR. Sharding: slice
(stamps[rank::size]).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from paraview.simple import Show, Render, SaveScreenshot, Delete

from swmf_vr import config
from swmf_vr.shading import write_colored_obj, write_color_assets
from swmf_vr.sharding import shard_slice, parse_leading_ints
from swmf_vr.io.seed_csv import is_header_only
from swmf_vr.paraview import tracing, preview
from swmf_vr.paraview.extract import extract_geometry_and_scalar


def parse_args(argv):
    """v6-compatible: positional ints first, then flags."""
    rank, size, max_files = 0, 1, None
    match = None
    shading_over = {}
    out_root, plt_dir = None, None
    ints, rest = parse_leading_ints(argv, 3)
    if len(ints) >= 2:
        rank, size = ints[0], ints[1]
    if len(ints) >= 3:
        max_files = ints[2]
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == '--match' and i + 1 < len(rest):
            match = rest[i + 1]; i += 2; continue
        if a == '--color-by' and i + 1 < len(rest):
            shading_over['scalar'] = rest[i + 1]; i += 2; continue
        if a == '--cmap' and i + 1 < len(rest):
            shading_over['cmap'] = rest[i + 1]; i += 2; continue
        if a == '--vmin' and i + 1 < len(rest):
            shading_over['vmin'] = float(rest[i + 1]); i += 2; continue
        if a == '--vmax' and i + 1 < len(rest):
            shading_over['vmax'] = float(rest[i + 1]); i += 2; continue
        if a == '--log':
            shading_over['log'] = True; i += 1; continue
        if a == '--no-log':
            shading_over['log'] = False; i += 1; continue
        if a == '--out-root' and i + 1 < len(rest):
            out_root = rest[i + 1]; i += 2; continue
        if a == '--plt-dir' and i + 1 < len(rest):
            plt_dir = rest[i + 1]; i += 2; continue
        i += 1
    return rank, size, max_files, match, shading_over, out_root, plt_dir


def main():
    rank, size, max_files, match, shading_over, out_root, plt_dir = parse_args(sys.argv[1:])

    cfg = config.default_config()
    if shading_over:
        cfg = cfg._replace(shading=cfg.shading._replace(**shading_over))
    if out_root:
        cfg = cfg._replace(paths=cfg.paths._replace(out_root=os.path.abspath(out_root)))
    if plt_dir:
        cfg = cfg._replace(paths=cfg.paths._replace(plt_dir=os.path.abspath(plt_dir)))

    scalar = cfg.shading.scalar
    valid = set(config.SEED_SCALARS) | set(config.POINT_SCALARS)
    if scalar not in valid:
        print("Unknown --color-by '{}'. Choose from: {}".format(
            scalar, ', '.join(sorted(valid))))
        sys.exit(1)
    log, vmin, vmax = config.resolve_color_range(cfg.shading)
    print('Color: scalar={} cmap={} range=[{:g},{:g}] {}'.format(
        scalar, cfg.shading.cmap, vmin, vmax, 'log' if log else 'linear'))

    out_root = cfg.paths.out_root
    seed_dirs = {s.name: os.path.join(out_root, s.seed_subdir) for s in cfg.surfaces}
    obj_dirs = {s.name: os.path.join(out_root, s.obj_subdir) for s in cfg.surfaces}
    for d in obj_dirs.values():
        os.makedirs(d, exist_ok=True)
    screenshot_dir = os.path.join(out_root, 'screenshot_renders')
    os.makedirs(screenshot_dir, exist_ok=True)

    render_view = tracing.make_render_view(cfg.camera)

    stamps = tracing.collect_timestamps(seed_dirs.values())
    if match is not None:
        stamps = [s for s in stamps if match in s]
    if not stamps:
        print('No seed CSVs found. Run Stage A first.')
        return
    mine = shard_slice(stamps, rank, size)
    if max_files is not None:
        mine = mine[:max_files]
    print('Total timestamps with seeds: {}'.format(len(stamps)))
    print('Worker {}/{} assigned {} timestamps.'.format(rank, size, len(mine)))

    for ts in mine:
        process_timestamp(cfg, ts, seed_dirs, obj_dirs, screenshot_dir,
                          render_view, scalar, log, vmin, vmax)
    print('Worker {}: done.'.format(rank))


def process_timestamp(cfg, ts, seed_dirs, obj_dirs, screenshot_dir,
                      render_view, scalar, log, vmin, vmax):
    plt_path = tracing.find_matching_plt(cfg.paths.plt_dir, ts)
    if not plt_path:
        print('[{}] no matching PLT in {} -- skipping'.format(ts, cfg.paths.plt_dir))
        return

    reader = b_calc = None
    proxies = []
    shown = False
    try:
        reader, b_calc = tracing.load_b_field(plt_path)
        for s in cfg.surfaces:
            csv_path = os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts))
            if not os.path.exists(csv_path):
                continue
            if is_header_only(csv_path):
                print('[{}] {}: no seeds -- skipping'.format(ts, s.name))
                continue

            out_path = os.path.join(obj_dirs[s.name], 'fieldlines_{}.obj'.format(ts))
            result = tracing.trace_streamlines(b_calc, csv_path, cfg.trace)
            if result is None:
                continue
            tracer, points_source, csv_reader, poly, seed_poly = result
            proxies.extend([tracer, points_source, csv_reader])

            points, cells, point_vals = extract_geometry_and_scalar(poly, seed_poly, scalar)
            umin, umax = write_colored_obj(points, cells, point_vals, out_path,
                                           scalar, log, vmin, vmax)
            write_color_assets(obj_dirs[s.name], cfg.shading.cmap, scalar, log, vmin, vmax)
            print('[{}] {}: {} cells -> {}  (U in [{:.3f},{:.3f}])'.format(
                ts, s.name, len(cells), os.path.basename(out_path), umin, umax))

            pv = preview.show_scalar_colored(render_view, poly, point_vals, cfg.shading)
            if pv is not None:
                proxies.append(pv)
            else:
                disp = Show(tracer, render_view)
                disp.DiffuseColor = list(s.rgb)
                disp.AmbientColor = list(s.rgb)
            shown = True

        if shown:
            Render()
            png_path = os.path.join(screenshot_dir, 'fieldlines_{}.png'.format(ts))
            SaveScreenshot(png_path, render_view)
            print('[{}] screenshot -> {}'.format(ts, os.path.basename(png_path)))
    except Exception as e:
        print('[{}] FAILED ({})'.format(ts, e))
    finally:
        for proxy in proxies:
            Delete(proxy)
        if b_calc is not None:
            Delete(b_calc)
        if reader is not None:
            Delete(reader)


if __name__ == '__main__':
    main()
