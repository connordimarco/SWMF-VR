#!/usr/bin/env python3
"""Full-plane slice movies with the shell mesh outline, for the VR team
(system python3, no pvbatch).

  python3 scripts/fp_planes_movie.py PLANE [rank size] [--band 2] [--fps 24]
                                     [--zlim RE] [--out-root DIR] [--ts-glob PAT]
  python3 scripts/fp_planes_movie.py PLANE --stitch      # frames -> mp4

PLANE is `noon-midnight` (right = azimuth 0 / noon, left = azimuth 180 / midnight)
or `dawn-dusk` (right = azimuth 90 / dusk, left = azimuth 270 / dawn). Each frame
joins the two opposite half-planes of run/fp_slices/npz/slices_<ts>.npz into one
meridional plane through the dipole axis (SM frame), on the fixed log colour scale,
and draws on it:

  solid white / cyan  the OUTER / INNER shell mesh cut by this plane. The mesh is
                      the same loft fp_cutaway.py builds (field lines of the delivered
                      obj_fp_*_gei OBJs, rotated GEI -> SM with sm_axes_gei_fp.csv,
                      resampled along arc length, neighbouring lines stitched in
                      foot-point order). This is the border a lofted VR mesh shows.
                      (--field-lines also dots the single field line at that azimuth,
                      projected; it leaves the mesh cut only at stretched times.)

Bz / AL / Dst strips with a time cursor sit below (sliceplot._draw_strips). Shard
over FRAMES with `rank size` (resumable: existing PNGs are skipped), then run
`--stitch` once -> run/fp_planes/<plane>.mp4.
"""

import os
import sys
import glob
import copy
import shutil
import argparse
import datetime
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swmf_vr import sliceplot
from fp_slices_movie import load_sm_axes, shell_lines_sm, loft_plane_cut, VMIN, VMAX, HANDOFF_RE

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PLANES = {
    'noon-midnight': dict(right=0.0, left=180.0, right_name='noon (MLT 12)', left_name='midnight (MLT 0)'),
    'dawn-dusk': dict(right=90.0, left=270.0, right_name='dusk (MLT 18)', left_name='dawn (MLT 6)'),
}
STYLE = (('outer', 'white'), ('inner', 'cyan'))


def half_plane_index(azimuths, az):
    k = np.where(np.isclose(azimuths, az))[0]
    if not len(k):
        raise ValueError('azimuth {} not in the slice cube ({})'.format(az, azimuths))
    return int(k[0])


def draw_outline(ax, items, name, col, az, sign, tol, first, field_lines=False):
    cut = loft_plane_cut(items, az)
    if len(cut):
        ax.plot(sign * cut[:, 0], cut[:, 1], color=col, lw=1.6, zorder=7,
                label='{} shell mesh cut by this plane'.format(name) if first else None)
    if not field_lines:
        return
    for foot_az, P in items:
        if abs((foot_az - az + 180.0) % 360.0 - 180.0) < tol:
            ax.plot(sign * np.hypot(P[:, 0], P[:, 1]), P[:, 2], color=col, lw=0.9, ls=':', zorder=7,
                    label='{} field line at this azimuth (projected)'.format(name) if first else None)
            break


def render_frame(plane, spec, rho, z, img_right, img_left, ts, label, unit, idx, out_png,
                 zlim, outlines, tol, field_lines=False):
    if idx is not None:
        fig = plt.figure(figsize=(12.0, 11.0))
        gs = fig.add_gridspec(4, 1, height_ratios=[7.5, 1, 1, 1], hspace=0.5)
        ax = fig.add_subplot(gs[0, 0])
        strip_axes = [fig.add_subplot(gs[r, 0]) for r in (1, 2, 3)]
    else:
        fig, ax = plt.subplots(figsize=(12.0, 7.0))
        strip_axes = []

    img = np.concatenate([img_left[:, ::-1], img_right[:, 1:]], axis=1)
    x = np.concatenate([-rho[::-1], rho[1:]])
    cmap = copy.copy(plt.get_cmap('inferno')); cmap.set_bad('#101015')
    shown = np.ma.masked_less_equal(img, 0.0)
    pm = ax.pcolormesh(x, z, np.clip(shown, VMIN, VMAX), cmap=cmap,
                       norm=LogNorm(vmin=VMIN, vmax=VMAX), shading='auto')
    for k, (name, col) in enumerate(STYLE):
        items = outlines.get(name, [])
        draw_outline(ax, items, name, col, spec['right'], +1, tol, True, field_lines)
        draw_outline(ax, items, name, col, spec['left'], -1, tol, False, field_lines)
    ax.add_patch(Circle((0, 0), 1.0, facecolor='#3a6ea5', edgecolor='none', zorder=5))
    ax.add_patch(Circle((0, 0), HANDOFF_RE, fill=False, color='0.6', lw=0.8, ls='--', zorder=6))
    ax.axhline(0, color='0.35', lw=0.5, zorder=4); ax.axvline(0, color='0.35', lw=0.5, zorder=4)
    ax.set_aspect('equal')
    ax.set_xlim(-rho[-1], rho[-1]); ax.set_ylim(-zlim, zlim)
    ax.set_xlabel(r'$\rho$ from the dipole axis (Re):  left = {}, right = {}'.format(
        spec['left_name'], spec['right_name']))
    ax.set_ylabel('z along the dipole axis (Re)')
    dt = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
    ax.set_title('{} plane through the dipole axis (SM frame)      {} UT      frame {}'.format(
        plane, dt.strftime('%Y-%m-%d %H:%M'), ts), fontsize=12)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.5)
    cb = fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('{} omnidirectional flux [{}]'.format(label, unit))
    if strip_axes:
        sliceplot._draw_strips(fig, strip_axes, idx, ts)
    fig.savefig(out_png, dpi=100, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plane', choices=sorted(PLANES))
    p.add_argument('shard', nargs='*', type=int, help='rank size (shard over frames)')
    p.add_argument('--band', type=int, default=2, help='0:<1 MeV 1:1-2 MeV 2:>2 MeV')
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--zlim', type=float, default=6.0)
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--indices', default=None)
    p.add_argument('--ts-glob', default='slices_*.npz')
    p.add_argument('--outline-tol', type=float, default=1.0)
    p.add_argument('--stitch', action='store_true', help='only stitch existing frames with ffmpeg')
    p.add_argument('--field-lines', action='store_true',
                   help='also dot the projected field line at each azimuth (off by default)')
    args = p.parse_args()
    rank, size = (args.shard + [0, 1])[:2] if args.shard else (0, 1)
    spec = PLANES[args.plane]

    frames_dir = os.path.join(args.out_root, 'fp_planes', 'frames_' + args.plane)
    movies_dir = os.path.join(args.out_root, 'fp_planes')
    os.makedirs(frames_dir, exist_ok=True)

    if args.stitch:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            print('no ffmpeg on PATH (module load ffmpeg)'); return
        out_mp4 = os.path.join(movies_dir, 'slice_{}_outlined.mp4'.format(args.plane))
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(args.fps),
               '-pattern_type', 'glob', '-i', os.path.join(frames_dir, 'frame_*.png'),
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', out_mp4]
        rc = subprocess.call(cmd)
        print('{} -> {}'.format('wrote' if rc == 0 else 'ffmpeg FAILED', out_mp4)); return

    npz_files = sorted(glob.glob(os.path.join(args.out_root, 'fp_slices', 'npz', args.ts_glob)))
    if not npz_files:
        print('no slices under', args.out_root); return
    first = np.load(npz_files[0], allow_pickle=True)
    azimuths = first['azimuths_deg']; rho = first['rho_re']; z = first['z_re']
    labels = list(first['labels']); quantity = str(first['quantity'])
    first.close()
    unit = 'cm$^{-2}$ s$^{-1}$' + ('' if quantity == 'omni' else ' sr$^{-1}$')
    i_r = half_plane_index(azimuths, spec['right']); i_l = half_plane_index(azimuths, spec['left'])
    indices = args.indices or os.path.join(args.out_root, 'slice_renders', 'indices.npz')
    idx = dict(np.load(indices, allow_pickle=True)) if os.path.exists(indices) else None
    sm_axes = load_sm_axes(os.path.join(args.out_root, 'sm_axes_gei_fp.csv'))

    mine = npz_files[rank::size]
    print('{} shard {}/{}: {} of {} frames'.format(args.plane, rank, size, len(mine), len(npz_files)))
    n_new = 0
    for f in mine:
        ts = os.path.basename(f)[len('slices_'):-len('.npz')]
        out_png = os.path.join(frames_dir, 'frame_{}.png'.format(ts))
        if os.path.exists(out_png):
            continue
        zf = np.load(f, allow_pickle=True)
        img_r = np.array(zf['cube'][i_r, :, :, args.band]); img_l = np.array(zf['cube'][i_l, :, :, args.band])
        zf.close()
        if ts not in sm_axes:
            raise KeyError('no SM axes row for {}'.format(ts))
        outlines = shell_lines_sm(args.out_root, ts, sm_axes[ts])
        render_frame(args.plane, spec, rho, z, img_r, img_l, ts, labels[args.band], unit, idx,
                     out_png, args.zlim, outlines, args.outline_tol, args.field_lines)
        n_new += 1
    print('{}: {} new frames in {}'.format(args.plane, n_new, frames_dir))


if __name__ == '__main__':
    main()
