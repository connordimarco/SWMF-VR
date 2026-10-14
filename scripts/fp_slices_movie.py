#!/usr/bin/env python3
"""One movie per azimuth slice, over the whole storm (system python3).

  python3 scripts/fp_slices_movie.py [rank size] [--band 2] [--fps 24]
                                     [--zlim RE] [--out-root DIR] [--no-stitch]

Reads run/fp_slices/npz/slices_<ts>.npz (from fp_slices.py) for every frame and,
for each azimuth in this shard, renders a frame per timestamp: the (rho, z)
flux cross-section on a fixed log colour scale (so the movie does not flicker),
the Earth, the 2.5 Re dipole/BATSRUS handoff as a dashed arc (so seam artefacts
are attributable), and the Bz / AL / Dst strips with a red cursor at the frame
time (sliceplot._draw_strips, from run/slice_renders/indices.npz). Then ffmpeg
stitches run/fp_slices/movie_frames/az<NNN>/ -> run/fp_slices/movies/slice_az<NNN>.mp4.
Resumable: existing frame PNGs are skipped.

Azimuth 0 deg is the SM +x direction (noon); MLT = az/15 + 12 (mod 24).

--outlines draws, on every frame, the inner/outer shell MESH cut by that
azimuth's half-plane (solid): the field lines of run/obj_fp_{outer,inner}_gei/
are rotated GEI -> SM with run/sm_axes_gei_fp.csv, lofted between neighbouring
lines exactly as fp_cutaway.py / the VR mesh does, and intersected with the
plane. The single field line whose north foot point lies within --outline-tol
deg of the azimuth is dotted for comparison (its apex can leave the meridian,
so the two differ at stretched times). Outputs go to movie_frames_outlined/
and movies_outlined/.
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
from matplotlib.patches import Wedge, Arc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from swmf_vr import sliceplot

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VMIN, VMAX = 1e3, 1e8          # >2 MeV omni, same scale as fp_slices.py / cutaway
HANDOFF_RE = 2.5


def load_sm_axes(path):
    """ts -> 3x3 matrix R with rows = SM unit vectors in GEI, so sm = R @ gei."""
    R = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('timestamp'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 11:
                continue
            v = np.array(parts[2:11], dtype=float)
            R[parts[0]] = np.vstack([v[0:3], v[3:6], v[6:9]])
    return R


LOFT_POINTS = 200          # arc-length resampling per line for the loft (fp_cutaway uses 160)
LOFT_MAX_GAP_DEG = 15.15   # neighbouring lines further apart than this are not stitched (1.01 h)


def resample_arc(P, n=LOFT_POINTS):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(3)])


def loft_plane_cut(items, az):
    """Intersection of the lofted shell (linear strips between neighbouring
    lines ordered by foot azimuth) with the half-plane at azimuth az, as
    (rho, z) points ordered north foot -> south foot. Empty if nothing crosses."""
    if not items:
        return np.zeros((0, 2))
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    A = [items[i][0] for i in order]
    S = [items[i][1] for i in order]
    a = np.radians(az)
    n = np.array([-np.sin(a), np.cos(a), 0.0]); u = np.array([np.cos(a), np.sin(a), 0.0])
    pts = []
    for i in range(len(S)):
        j = (i + 1) % len(S)
        if (A[j] - A[i]) % 360.0 > LOFT_MAX_GAP_DEG:
            continue
        di, dj = S[i] @ n, S[j] @ n
        for k in range(len(S[i])):
            if di[k] * dj[k] <= 0.0 and di[k] != dj[k]:
                w = di[k] / (di[k] - dj[k])
                p = S[i][k] * (1.0 - w) + S[j][k] * w
                if p @ u > 0.0:
                    pts.append((k, np.hypot(p[0], p[1]), p[2]))
    pts.sort(key=lambda t: t[0])
    return np.array([(r, z) for _, r, z in pts]) if pts else np.zeros((0, 2))


def shell_lines_sm(out_root, ts, R):
    """{'outer': [...], 'inner': [...]}: each line as (foot_az_deg, P) with P the
    resampled 3-D polyline in the SM/slice frame, from the delivered GEI OBJs."""
    out = {}
    for name in ('outer', 'inner'):
        path = os.path.join(out_root, 'obj_fp_{}_gei'.format(name), 'fieldlines_{}.obj'.format(ts))
        verts, lines = [], []
        if os.path.exists(path):
            with open(path) as f:
                for ln in f:
                    if ln.startswith('v '):
                        verts.append(ln.split()[1:4])
                    elif ln.startswith('l '):
                        lines.append([int(t.split('/')[0]) - 1 for t in ln.split()[1:]])
        items = []
        if verts:
            V = np.array(verts, dtype=float) @ R.T          # GEI -> SM
            for idx in lines:
                P = V[idx]
                foot_az = np.degrees(np.arctan2(P[0, 1], P[0, 0])) % 360.0   # north foot (vertex 0)
                items.append((foot_az, resample_arc(P)))
        out[name] = items
    return out


OUTLINE_STYLE = (('outer', 'white', 'outer shell mesh'),
                 ('inner', 'cyan', 'inner shell mesh'))


def render_frame(az, mlt, rho, z, img, ts, label, unit, idx, out_png, zlim,
                 outlines=None, outline_tol=1.0):
    if idx is not None:
        fig = plt.figure(figsize=(9.0, 10.5))
        gs = fig.add_gridspec(4, 1, height_ratios=[7, 1, 1, 1], hspace=0.5)
        ax = fig.add_subplot(gs[0, 0])
        strip_axes = [fig.add_subplot(gs[r, 0]) for r in (1, 2, 3)]
    else:
        fig, ax = plt.subplots(figsize=(9.0, 7.0))
        strip_axes = []

    cmap = copy.copy(plt.get_cmap('inferno')); cmap.set_bad('#101015')
    shown = np.ma.masked_less_equal(img, 0.0)
    pm = ax.pcolormesh(rho, z, np.clip(shown, VMIN, VMAX), cmap=cmap,
                       norm=LogNorm(vmin=VMIN, vmax=VMAX), shading='auto')
    ax.add_patch(Wedge((0, 0), 1.0, -90, 90, facecolor='#3a6ea5', edgecolor='none', zorder=5))
    ax.add_patch(Arc((0, 0), 2 * HANDOFF_RE, 2 * HANDOFF_RE, theta1=-90, theta2=90,
                     color='0.6', lw=0.8, ls='--', zorder=6))
    ax.axhline(0, color='0.35', lw=0.5, zorder=4)
    if outlines is not None:
        for name, col, lab in OUTLINE_STYLE:
            cut = loft_plane_cut(outlines.get(name, []), az)
            if len(cut):
                ax.plot(cut[:, 0], cut[:, 1], color=col, lw=1.6, zorder=7,
                        label='{} cut by this plane'.format(lab))
            first = True
            for foot_az, P in outlines.get(name, []):
                if abs((foot_az - az + 180.0) % 360.0 - 180.0) < outline_tol:
                    ax.plot(np.hypot(P[:, 0], P[:, 1]), P[:, 2], color=col, lw=0.9, ls=':', zorder=7,
                            label='{} field line at this azimuth (projected)'.format(name) if first else None)
                    first = False
        ax.legend(loc='upper right', fontsize=7, framealpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(rho[0], rho[-1]); ax.set_ylim(-zlim, zlim)
    ax.set_xlabel(r'$\rho$ from dipole axis (Re)'); ax.set_ylabel('z along dipole axis (Re)')
    dt = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
    ax.set_title('Slice at {:.0f}$^\\circ$ about the dipole axis  (MLT {:04.1f})     '
                 '{} UT'.format(az, mlt, dt.strftime('%Y-%m-%d %H:%M')), fontsize=12)
    cb = fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('{} omnidirectional flux [{}]'.format(label, unit))
    if strip_axes:
        sliceplot._draw_strips(fig, strip_axes, idx, ts)
    fig.savefig(out_png, dpi=100, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('shard', nargs='*', type=int, help='rank size (shard over azimuths)')
    p.add_argument('--band', type=int, default=2, help='0:<1 MeV 1:1-2 MeV 2:>2 MeV')
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--zlim', type=float, default=6.0, help='|z| shown (Re)')
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--indices', default=None,
                   help='indices .npz for the Bz/AL/Dst strips (default run/slice_renders/indices.npz)')
    p.add_argument('--bmin-tol', type=float, default=None,
                   help='mask pixels whose local |B| / RBE B_min is below this '
                        '(e.g. 0.8): the BATSRUS and RBE field-line mappings '
                        'disagree there. Default: no mask (equatorial-flux clamp)')
    p.add_argument('--no-stitch', action='store_true')
    p.add_argument('--outlines', action='store_true',
                   help='draw the in-plane inner/outer shell field lines (obj_fp_*_gei + sm_axes_gei_fp.csv)')
    p.add_argument('--outline-tol', type=float, default=1.0,
                   help='max |foot azimuth - slice azimuth| in deg for a line to be drawn')
    p.add_argument('--ts-glob', default='slices_*.npz',
                   help="npz file pattern, e.g. 'slices_20240511_09*.npz' for a subset")
    args = p.parse_args()
    rank, size = (args.shard + [0, 1])[:2] if args.shard else (0, 1)

    npz_files = sorted(glob.glob(os.path.join(args.out_root, 'fp_slices', 'npz', args.ts_glob)))
    if not npz_files:
        print('no slices_*.npz under', args.out_root); return
    first = np.load(npz_files[0], allow_pickle=True)
    azimuths = first['azimuths_deg']; rho = first['rho_re']; z = first['z_re']
    labels = list(first['labels']); quantity = str(first['quantity'])
    unit = 'cm$^{-2}$ s$^{-1}$' + ('' if quantity == 'omni' else ' sr$^{-1}$')
    first.close()
    indices = args.indices or os.path.join(args.out_root, 'slice_renders', 'indices.npz')
    idx = dict(np.load(indices, allow_pickle=True)) if os.path.exists(indices) else None

    my_az = [i for i in range(len(azimuths)) if i % size == rank]
    tag = '' if args.bmin_tol is None else '_bmin{:g}'.format(args.bmin_tol)
    sm_axes = None
    if args.outlines:
        tag += '_outlined'
        sm_axes = load_sm_axes(os.path.join(args.out_root, 'sm_axes_gei_fp.csv'))
    frames_root = os.path.join(args.out_root, 'fp_slices', 'movie_frames' + tag)
    movies_dir = os.path.join(args.out_root, 'fp_slices', 'movies' + tag)
    os.makedirs(movies_dir, exist_ok=True)
    print('shard {}/{}: azimuths {} over {} frames'.format(
        rank, size, [int(azimuths[i]) for i in my_az], len(npz_files)))

    for i in my_az:
        az = float(azimuths[i]); mlt = (az / 15.0 + 12.0) % 24.0
        fdir = os.path.join(frames_root, 'az{:03d}'.format(int(round(az))))
        os.makedirs(fdir, exist_ok=True)
        n_new = 0
        for f in npz_files:
            ts = os.path.basename(f)[len('slices_'):-len('.npz')]
            out_png = os.path.join(fdir, 'frame_{}.png'.format(ts))
            if os.path.exists(out_png):
                continue
            zf = np.load(f, allow_pickle=True)
            img = np.array(zf['cube'][i, :, :, args.band])
            if args.bmin_tol is not None and 'bratio' in zf.files:
                br = zf['bratio'][i]
                img[(br > 0) & (br < args.bmin_tol)] = 0.0
            zf.close()
            outlines = None
            if sm_axes is not None:
                if ts not in sm_axes:
                    raise KeyError('no SM axes row for {} in sm_axes_gei_fp.csv'.format(ts))
                outlines = shell_lines_sm(args.out_root, ts, sm_axes[ts])
            render_frame(az, mlt, rho, z, img, ts, labels[args.band], unit, idx, out_png, args.zlim,
                         outlines=outlines, outline_tol=args.outline_tol)
            n_new += 1
        print('az {:03d}: {} new frames in {}'.format(int(az), n_new, fdir))
        if args.no_stitch:
            continue
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            print('  no ffmpeg on PATH (module load ffmpeg); frames kept'); continue
        out_mp4 = os.path.join(movies_dir, 'slice_az{:03d}.mp4'.format(int(round(az))))
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(args.fps),
               '-pattern_type', 'glob', '-i', os.path.join(fdir, 'frame_*.png'),
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', out_mp4]
        rc = subprocess.call(cmd)
        print('  {} -> {}'.format('wrote' if rc == 0 else 'ffmpeg FAILED', out_mp4))


if __name__ == '__main__':
    main()
