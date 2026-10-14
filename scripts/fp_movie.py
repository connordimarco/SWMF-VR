#!/usr/bin/env python3
"""Render movie frames from the foot-point field-line OBJs (system python3).

  python3 scripts/fp_movie.py [rank size] [--out-root DIR] [--force]

Reads the three obj_fp_* directories, draws each frame's shells in a fixed
side view (GSM X-Z, the same viewpoint the old pipeline screenshots used),
and writes one PNG per timestamp to run/fp_movie_frames/. Frames whose PNG
already exists are skipped, so it shards and resumes like everything else.

The OBJs already contain the full hybrid lines (dipole below 2.5 Re,
BATSRUS above), so this is a pure drawing pass -- no tracing, no ParaView.
Colors are the surface colors from config (inner green, outer orange,
danger red).

Assemble the movie afterwards with ffmpeg (module load ffmpeg):

  ffmpeg -framerate 30 -pattern_type glob -i 'run/fp_movie_frames/frame_*.png' \
         -c:v libx264 -pix_fmt yuv420p run/fp_fieldlines.mp4
"""

import os
import re
import sys
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import config


def read_obj_lines(path):
    """The OBJ's polylines as a list of (n_points x 3) arrays."""
    verts, lines = [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                verts.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('l '):
                lines.append([int(t.split('/')[0]) - 1 for t in ln.split()[1:]])
    verts = np.array(verts)
    return [verts[c] for c in lines]


def collect_timestamps(obj_dirs):
    stamps = set()
    for d in obj_dirs.values():
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = re.match(r'fieldlines_(\d{8}_\d{6})\.obj', fname)
            if m:
                stamps.add(m.group(1))
    return sorted(stamps)


def render_frame(ts, obj_dirs, surfaces, out_png):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    for s in surfaces:
        path = os.path.join(obj_dirs[s.name], 'fieldlines_{}.obj'.format(ts))
        if not os.path.exists(path):
            continue
        for line in read_obj_lines(path):
            ax.plot(line[:, 0], line[:, 2], lw=0.6, color=s.rgb, alpha=0.65)

    # Earth (1 Re) and the dipole/BATSRUS handoff sphere for scale.
    circle_angle = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.fill(np.cos(circle_angle), np.sin(circle_angle), color='#3a6ea5', zorder=5)
    ax.plot(2.5 * np.cos(circle_angle), 2.5 * np.sin(circle_angle),
            ':', color='#666666', lw=0.8)

    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.set_xlabel('GSM X (Re)', color='white')
    ax.set_ylabel('GSM Z (Re)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#555555')
    pretty = '{}-{}-{} {}:{} UT'.format(ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13])
    ax.set_title('Radiation belt boundary field lines -- ' + pretty,
                 color='white', fontsize=14)

    handles = [plt.Line2D([0], [0], color=s.rgb, lw=2,
                          label={'inner': 'inner (flux 10)',
                                 'outer': 'outer (flux 10, from last closed)',
                                 'danger': 'danger (flux peak)'}[s.name])
               for s in surfaces]
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              facecolor='black', edgecolor='#555555', labelcolor='white')

    fig.tight_layout()
    fig.savefig(out_png, facecolor='black')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('shard', nargs='*', type=int)
    p.add_argument('--out-root', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    rank, size = (args.shard + [0, 1])[:2]

    cfg = config.CRAConfig()
    out = os.path.abspath(args.out_root) if args.out_root else cfg.paths.out_root
    obj_dirs = {s.name: os.path.join(out, 'obj_fp_' + s.name)
                for s in cfg.surfaces}
    frame_dir = os.path.join(out, 'fp_movie_frames')
    os.makedirs(frame_dir, exist_ok=True)

    stamps = collect_timestamps(obj_dirs)
    mine = stamps[rank::size]
    print('{} stamps total; worker {}/{} -> {}'.format(len(stamps), rank, size, len(mine)))

    done = 0
    for ts in mine:
        out_png = os.path.join(frame_dir, 'frame_{}.png'.format(ts))
        if os.path.exists(out_png) and not args.force:
            continue
        render_frame(ts, obj_dirs, cfg.surfaces, out_png)
        done += 1
        if done % 50 == 0:
            print('  worker {}: {} rendered (latest {})'.format(rank, done, ts))
    print('worker {}: {} frames rendered'.format(rank, done))


if __name__ == '__main__':
    main()
