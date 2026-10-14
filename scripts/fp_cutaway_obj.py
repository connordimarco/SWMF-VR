#!/usr/bin/env python3
"""Export the cutaway (half donut + shaded cut face) as OBJ for the VR team
(pvbatch).

  "$PV_BATCH" scripts/fp_cutaway_obj.py --ts 20240510_130000 [--band 2]
       [--frame gei|gsm] [--resolution 0.025] [--out-dir DIR]

Writes, per frame, into <out-dir>/<ts>/ :
  cutaway_<ts>.obj          three groups: g outer (half skin), g inner (hole
                            wall), g cutface (the flux-shaded meridian face).
                            The face carries a per-vertex `vt` whose u is the
                            normalised log10 flux, sampled against ramp.png --
                            the same convention as the field-line OBJs.
  cutaway_<ts>_vcolor.obj   same geometry, but the face uses `v x y z r g b`
                            vertex colours (non-standard OBJ extension that
                            MeshLab / Blender / three.js read) for viewers
                            without MTL texture support.
  material.mtl, ramp.png    skin materials (flat colours) + the flux ramp.
  colormap.json             how u maps back to flux (log, vmin, vmax).

Geometry is built exactly as in fp_cutaway.py (lofted shells clipped to the
dusk half y >= 0, cut face on the y = 0 plane), then rotated GSM -> GEI with
the frame's geopack transform when --frame gei (default).
"""

import os
import sys
import json
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fp_cutaway as fc          # loft, mask, shading, geopack fixup
from paraview.simple import Clip, ExtractSurface
from paraview import servermanager as _sm

from swmf_vr import config, render, dipole, frames
import swmf_vr.data as data
from swmf_vr.color import scalar_to_u

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SKIN_COLORS = {'outer': (0.95, 0.55, 0.12), 'inner': (0.85, 0.45, 0.10)}


def half_surface_mesh(lines):
    """Loft the lines and keep the dusk half (y >= 0). Returns (points, tris)
    as numpy arrays -- points (N,3), tris (M,3) 0-based."""
    producer = fc.loft_surface(lines, 'Loft')
    clip = Clip(Input=producer)
    clip.ClipType = 'Plane'
    clip.ClipType.Origin = [0.0, 0.0, 0.0]
    clip.ClipType.Normal = [0.0, 1.0, 0.0]
    clip.Invert = 0
    surface = ExtractSurface(Input=clip)
    surface.UpdatePipeline()
    poly = _sm.Fetch(surface)
    points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
    tris = []
    for c in range(poly.GetNumberOfCells()):
        ids = poly.GetCell(c).GetPointIds()
        n = ids.GetNumberOfIds()
        if n == 3:
            tris.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])
        elif n > 3:                                   # fan-triangulate anything bigger
            for k in range(1, n - 1):
                tris.append([ids.GetId(0), ids.GetId(k), ids.GetId(k + 1)])
    return points, np.array(tris, dtype=int)


def cut_face_mesh(xs, zs, image, flux_min, flux_max):
    """Triangles of the y = 0 grid cells whose four corners all have flux.
    Returns (points (N,3), tris (M,3), u (N,) colour coordinate, flux (N,))."""
    n_z, n_x = image.shape
    index = -np.ones((n_z, n_x), dtype=int)
    points, u_vals, flux_vals = [], [], []
    for iz in range(n_z):
        for ix in range(n_x):
            if np.isfinite(image[iz, ix]):
                index[iz, ix] = len(points)
                points.append((xs[ix], 0.0, zs[iz]))
                u_vals.append(scalar_to_u(float(image[iz, ix]), True, flux_min, flux_max))
                flux_vals.append(float(image[iz, ix]))
    tris = []
    for iz in range(n_z - 1):
        for ix in range(n_x - 1):
            a, b = index[iz, ix], index[iz, ix + 1]
            c, d = index[iz + 1, ix], index[iz + 1, ix + 1]
            if min(a, b, c, d) < 0:
                continue
            tris.append([a, b, d]); tris.append([a, d, c])
    return (np.array(points), np.array(tris, dtype=int),
            np.array(u_vals), np.array(flux_vals))


def inferno_rgb(u):
    import matplotlib.cm as cm
    return cm.get_cmap('inferno')(np.clip(u, 0.0, 1.0))[:, :3]


def write_obj(path, groups, use_vertex_colors=False):
    """groups: list of (name, material, points, tris, u or None, rgb or None).
    Vertex/texcoord indices are global and 1-based in OBJ."""
    with open(path, 'w') as f:
        f.write('# radiation belt cutaway: half donut (outer skin, inner hole wall) + flux-shaded cut face\n')
        f.write('mtllib material.mtl\n')
        v_offset = 0
        vt_offset = 0
        for name, material, points, tris, u, rgb in groups:
            f.write('g {}\nusemtl {}\n'.format(name, material))
            for i, (x, y, z) in enumerate(points):
                if use_vertex_colors and rgb is not None:
                    f.write('v {:.5f} {:.5f} {:.5f} {:.4f} {:.4f} {:.4f}\n'.format(
                        x, y, z, rgb[i, 0], rgb[i, 1], rgb[i, 2]))
                else:
                    f.write('v {:.5f} {:.5f} {:.5f}\n'.format(x, y, z))
            has_vt = (u is not None) and not use_vertex_colors
            if has_vt:
                for val in u:
                    f.write('vt {:.5f} 0.5\n'.format(val))
            for a, b, c in tris:
                if has_vt:
                    f.write('f {0}/{3} {1}/{4} {2}/{5}\n'.format(
                        a + 1 + v_offset, b + 1 + v_offset, c + 1 + v_offset,
                        a + 1 + vt_offset, b + 1 + vt_offset, c + 1 + vt_offset))
                else:
                    f.write('f {} {} {}\n'.format(a + 1 + v_offset, b + 1 + v_offset, c + 1 + v_offset))
            v_offset += len(points)
            if has_vt:
                vt_offset += len(points)


def write_materials(out_dir, flux_min, flux_max):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ramp = inferno_rgb(np.linspace(0.0, 1.0, 256))
    img = np.repeat((ramp * 255).astype('uint8')[None, :, :], 8, axis=0)
    plt.imsave(os.path.join(out_dir, 'ramp.png'), img)
    with open(os.path.join(out_dir, 'material.mtl'), 'w') as f:
        for name, (r, g, b) in SKIN_COLORS.items():
            f.write('newmtl skin_{}\nKa {:.3f} {:.3f} {:.3f}\nKd {:.3f} {:.3f} {:.3f}\n'
                    'Ks 0.3 0.3 0.3\nNs 40\nillum 2\n\n'.format(name, r * 0.3, g * 0.3, b * 0.3, r, g, b))
        f.write('newmtl ramp\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd ramp.png\n')
    with open(os.path.join(out_dir, 'colormap.json'), 'w') as f:
        json.dump({'scalar': 'omnidirectional flux, > 2 MeV (cm^-2 s^-1)',
                   'cmap': 'inferno', 'vmin': flux_min, 'vmax': flux_max, 'log': True,
                   'ramp': 'ramp.png',
                   'mapping': 'vt u = (log10(flux) - log10(vmin)) / (log10(vmax) - log10(vmin)); U=0 -> vmin, U=1 -> vmax'},
                  f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True)
    p.add_argument('--band', type=int, default=2)
    p.add_argument('--frame', choices=['gei', 'gsm'], default='gei')
    p.add_argument('--resolution', type=float, default=0.025)
    p.add_argument('--extent', type=float, default=10.0)
    p.add_argument('--flux-min', type=float, default=1e3)
    p.add_argument('--flux-max', type=float, default=fc.FLUX_MAX)
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    ts = args.ts
    out_dir = args.out_dir or os.path.join(args.out_root, 'cutaway_obj_' + args.frame, ts)
    os.makedirs(out_dir, exist_ok=True)

    cfg = config.CRAConfig()
    frame_dt = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
    frames.recalc_for(frame_dt)
    t = data.frame_index_for(cfg.paths.npz_path, frame_dt, data.base_datetime(cfg.time.base_iso))
    if t is None:
        sys.exit('[{}] no flux frame at this time in {}'.format(ts, cfg.paths.npz_path))
    plt_path = render.find_matching_plt(cfg.paths.plt_dir, ts)

    outer_lines = fc.read_obj_polylines(os.path.join(args.out_root, 'obj_fp_outer',
                                                     'fieldlines_{}.obj'.format(ts)))
    inner_lines = fc.read_obj_polylines(os.path.join(args.out_root, 'obj_fp_inner',
                                                     'fieldlines_{}.obj'.format(ts)))

    # --- geometry in GSM ---------------------------------------------------------
    outer_pts, outer_tris = half_surface_mesh(outer_lines)
    inner_pts, inner_tris = half_surface_mesh(inner_lines)
    xs, zs, image = fc.shade_cut_plane(
        plt_path, cfg.paths.npz_path, t, args.band, args.resolution, args.extent,
        dipole.HANDOFF_RADIUS_RE,
        inside_fn=lambda xz: fc.cross_section_mask(outer_lines, inner_lines, xz))
    image[np.isfinite(image) & (image < args.flux_min)] = args.flux_min
    face_pts, face_tris, face_u, face_flux = cut_face_mesh(xs, zs, image, args.flux_min, args.flux_max)
    print('[{}] outer {} tris, inner {} tris, cut face {} verts / {} tris'.format(
        ts, len(outer_tris), len(inner_tris), len(face_pts), len(face_tris)))

    # --- rotate to GEI if asked -------------------------------------------------
    if args.frame == 'gei':
        def to_gei(points):
            return np.array([frames.gsm_to_gei(float(x), float(y), float(z)) for x, y, z in points])
        outer_pts, inner_pts, face_pts = to_gei(outer_pts), to_gei(inner_pts), to_gei(face_pts)

    # --- write -------------------------------------------------------------------
    face_rgb = inferno_rgb(face_u)
    groups = [('outer', 'skin_outer', outer_pts, outer_tris, None, None),
              ('inner', 'skin_inner', inner_pts, inner_tris, None, None),
              ('cutface', 'ramp', face_pts, face_tris, face_u, face_rgb)]
    write_obj(os.path.join(out_dir, 'cutaway_{}.obj'.format(ts)), groups)
    write_obj(os.path.join(out_dir, 'cutaway_{}_vcolor.obj'.format(ts)), groups,
              use_vertex_colors=True)
    write_materials(out_dir, args.flux_min, args.flux_max)
    with open(os.path.join(out_dir, 'README.txt'), 'w') as f:
        f.write('Radiation belt cutaway, {} UT, frame = {}\n'.format(
            frame_dt.strftime('%Y-%m-%d %H:%M'), args.frame.upper()))
        f.write('Units: Re. Groups: outer (half skin, y_GSM >= 0 dusk half), inner (hole wall), '
                'cutface (y_GSM = 0 noon-midnight plane).\n')
        f.write('cutface colour: vt u against ramp.png, log10 flux {:g}..{:g} '
                '(omnidirectional, > 2 MeV). See colormap.json.\n'.format(args.flux_min, args.flux_max))
        f.write('cutaway_*_vcolor.obj: identical geometry with v x y z r g b vertex colours instead.\n')
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
