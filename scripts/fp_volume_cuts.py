#!/usr/bin/env python3
"""Cuts through the 3D flux volume (system python3, matplotlib).

  # one PNG per frame for a movie (sharded, resumable):
  python3 scripts/fp_volume_cuts.py movie [rank size] [--band 2] [--out-root DIR]

  # one figure comparing several times (columns) x the three cuts (rows):
  python3 scripts/fp_volume_cuts.py phases --ts TS1 TS2 ... [--band 2] [--out PNG]

The volume lattice is regular (0.5 Re, +-10 Re, GSM) and contains the
planes x = 0, y = 0 and z = 0 exactly, so a "cut" is just the lattice points
lying in that plane, shown as an image (one pixel per 0.5 Re voxel; empty
voxels -- open field lines or off the RBE grid -- are left blank).

The three cuts:
  y = 0 : noon-midnight meridian (X-Z)  -- Sun to the right, north up
  z = 0 : GSM equatorial plane   (X-Y)  -- Sun to the right, dusk up
  x = 0 : dawn-dusk meridian     (Y-Z)  -- dusk to the right, north up
Note GSM z is NOT the dipole axis (tilt ~15-25 deg over this run), so the
belt looks tilted in the meridian cuts; that is real, see dipole.py.
"""

import os
import re
import sys
import glob
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BAND_LABELS = ['< 1 MeV', '1-2 MeV', '> 2 MeV']
# Log color range. This is OMNIDIRECTIONAL flux (integrated over 4 pi sr), so
# it runs ~1e5 x higher than the equatorial directional flux in the 2D plots:
# the >2 MeV core reaches ~1e6 quiet and ~1e8 at the storm peak.
FLUX_MIN, FLUX_MAX = 1e1, 1e7

CUTS = [  # (title, index of the coordinate held at 0, horizontal axis, vertical axis, labels)
    ('y = 0  (noon-midnight meridian)', 1, 0, 2, 'GSM X (Re)', 'GSM Z (Re)'),
    ('z = 0  (equatorial plane)',       2, 0, 1, 'GSM X (Re)', 'GSM Y (Re)'),
    ('x = 0  (dawn-dusk meridian)',     0, 1, 2, 'GSM Y (Re)', 'GSM Z (Re)'),
]


def load_volume(npz_path, band):
    z = np.load(npz_path, allow_pickle=True)
    points = np.asarray(z['points'], dtype=float)
    flux = np.asarray(z['flux'], dtype=float)[:, band]
    step = float(z['step']); extent = float(z['extent'])
    return points, flux, step, extent


def cut_image(points, flux, step, extent, plane_axis, h_axis, v_axis):
    """2D image of the voxels lying in the plane coord[plane_axis] = 0.
    Rows are the vertical axis (ascending), columns the horizontal one."""
    n = int(round(2 * extent / step)) + 1
    image = np.full((n, n), np.nan)
    in_plane = np.abs(points[:, plane_axis]) < step / 2.0
    for k in np.where(in_plane)[0]:
        col = int(round((points[k, h_axis] + extent) / step))
        row = int(round((points[k, v_axis] + extent) / step))
        if 0 <= row < n and 0 <= col < n:
            image[row, col] = flux[k]
    return image


def draw_cut(ax, image, extent, step, title, xlabel, ylabel, show_labels=True):
    shown = np.where(np.isfinite(image) & (image > 0), image, np.nan)
    half = extent + step / 2.0
    im = ax.imshow(shown, origin='lower', extent=[-half, half, -half, half],
                   cmap='inferno', norm=LogNorm(vmin=FLUX_MIN, vmax=FLUX_MAX),
                   interpolation='nearest')
    angle = np.linspace(0.0, 2.0 * np.pi, 100)
    ax.fill(np.cos(angle), np.sin(angle), color='#3a6ea5', zorder=5)
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_title(title, fontsize=10, color='white')
    if show_labels:
        ax.set_xlabel(xlabel, color='white'); ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#555555')
    return im


def pretty_time(ts):
    return '{}-{}-{} {}:{} UT'.format(ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13])


def render_movie_frame(npz_path, ts, band, out_png):
    points, flux, step, extent = load_volume(npz_path, band)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), dpi=110)
    fig.patch.set_facecolor('black')
    im = None
    for ax, (title, plane_axis, h_axis, v_axis, xl, yl) in zip(axes, CUTS):
        image = cut_image(points, flux, step, extent, plane_axis, h_axis, v_axis)
        im = draw_cut(ax, image, extent, step, title, xl, yl)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('omnidirectional flux, {} (cm$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$)'
                   .format(BAND_LABELS[band]), color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    fig.suptitle('Radiation belt flux volume -- ' + pretty_time(ts),
                 color='white', fontsize=14)
    fig.savefig(out_png, facecolor='black')
    plt.close(fig)


def render_phases(npz_dir, stamps, band, out_png):
    fig, axes = plt.subplots(3, len(stamps), figsize=(4.6 * len(stamps), 13), dpi=110)
    fig.patch.set_facecolor('black')
    im = None
    for col, ts in enumerate(stamps):
        npz_path = os.path.join(npz_dir, 'volume_{}_bands.npz'.format(ts))
        points, flux, step, extent = load_volume(npz_path, band)
        for row, (title, plane_axis, h_axis, v_axis, xl, yl) in enumerate(CUTS):
            image = cut_image(points, flux, step, extent, plane_axis, h_axis, v_axis)
            ax = axes[row][col]
            im = draw_cut(ax, image, extent, step,
                          pretty_time(ts) if row == 0 else '', xl, yl,
                          show_labels=(row == 2 or col == 0))
            if col == 0:
                ax.text(-0.32, 0.5, title.split('(')[1].rstrip(')'), transform=ax.transAxes,
                        rotation=90, va='center', ha='center', color='white', fontsize=11)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('omnidirectional flux, {} (cm$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$)'
                   .format(BAND_LABELS[band]), color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    fig.suptitle('Radiation belt flux volume through the storm (foot-point method)',
                 color='white', fontsize=15)
    fig.savefig(out_png, facecolor='black')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['movie', 'phases'])
    p.add_argument('shard', nargs='*', type=int)
    p.add_argument('--band', type=int, default=2, help='0: <1 MeV, 1: 1-2 MeV, 2: >2 MeV')
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--ts', nargs='*', default=[])
    p.add_argument('--out', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    npz_dir = os.path.join(args.out_root, 'volume_renders_fp', 'npz')

    if args.mode == 'phases':
        out = args.out or os.path.join(args.out_root, 'fp_volume_phases.png')
        render_phases(npz_dir, args.ts, args.band, out)
        print('saved', out)
        return

    rank, size = (args.shard + [0, 1])[:2]
    frame_dir = os.path.join(args.out_root, 'fp_volume_frames')
    os.makedirs(frame_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(npz_dir, 'volume_*_bands.npz')))
    mine = paths[rank::size]
    print('{} volumes; worker {}/{} -> {}'.format(len(paths), rank, size, len(mine)))
    done = 0
    for npz_path in mine:
        ts = re.search(r'volume_(\d{8}_\d{6})_bands', npz_path).group(1)
        out_png = os.path.join(frame_dir, 'volcut_{}.png'.format(ts))
        if os.path.exists(out_png) and not args.force:
            continue
        render_movie_frame(npz_path, ts, args.band, out_png)
        done += 1
    print('worker {}: {} frames rendered'.format(rank, done))


if __name__ == '__main__':
    main()
