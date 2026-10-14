#!/usr/bin/env python3
"""Cutaway render: the belt as a solid donut built from the field lines,
cut in half at the noon-midnight meridian, with the flux shaded on the cut
face (pvbatch).

  "$PV_BATCH" scripts/fp_cutaway.py --ts 20240511_094100 [--band 2]
       [--resolution 0.1] [--flux-min 1e3] [--out PNG] [--out-root DIR]

How it is built
---------------
1. SURFACES. The outer shell OBJ (one field line per MLT sector) is lofted
   into a surface: each line is resampled to the same number of points
   along its arc length (north foot -> south foot), and neighbouring MLT
   lines are stitched with quad strips. Same for the inner shell, which is
   the wall of the donut hole. Both are clipped to the dusk half (y >= 0).

2. CUT FACE. A fine grid on the y = 0 plane is shaded with the flux from
   the SAME foot-point mapping the volume uses (footpoint_volume.py):
   outside the 2.5 Re handoff sphere the point's foot point comes from
   BATSRUS (theta_1 / phi_1), inside it from the analytic dipole -- so the
   belt core close to Earth is filled, unlike the voxel lattice which stops
   at rBody. The RBE distribution at that foot point is Liouville-mapped to
   the local |B| (mapping.py), giving omnidirectional flux.

3. MASK. The cut face is shaded only inside the donut's cross-section: the
   region between the outer and inner shells' own noon and midnight arcs
   (the lines of the OBJs lying in the cut plane), so the painted region
   ends exactly where the lofted surface does.

Colour range is the volume's (omnidirectional flux); Sun to the right,
north up, camera on the dawn side looking at the cut face.
"""

import os
import sys
import math
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# geopack lives in the system python's user site; borrow it (pure python).
try:
    from geopack import geopack
except (ModuleNotFoundError, ValueError):
    import glob as _glob
    for _site_dir in sorted(_glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages'))):
        if not os.path.isdir(os.path.join(_site_dir, 'geopack')):
            continue
        sys.path.append(_site_dir)
        try:
            from geopack import geopack
            break
        except Exception:
            sys.path.remove(_site_dir)
            for _mod in [m for m in sys.modules if m.startswith('geopack')]:
                del sys.modules[_mod]

from paraview.simple import (
    TrivialProducer, Clip, ExtractSurface, GenerateSurfaceNormals, Sphere, Show, Render,
    SaveScreenshot, CreateView, GetColorTransferFunction, GetScalarBar,
    ColorBy, Text, Line, Cone, Transform, a3DText, _DisableFirstRenderCameraReset,
)
from paraview import vtk as pvtk
from paraview.vtk.util import numpy_support
from matplotlib.path import Path as PolygonPath

from swmf_vr import config, render, mapping, dipole, frames
import swmf_vr.data as data
from slice3d import BANDS
import footpoint_volume as fpv

_DisableFirstRenderCameraReset()
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

FLUX_MAX = 1e8                 # the > 2 MeV core reaches ~1e8 at the storm peak
BAND_LABELS = ['< 1 MeV', '1-2 MeV', '> 2 MeV']
POINTS_PER_LINE = 160          # resampling count for the loft
EARTH_DIPOLE_TESLA = 3.03e-5   # B0 moment seen in the RBE data (bo * ro^3)


# --- OBJ lines -> lofted surface ---------------------------------------------

def read_obj_polylines(path):
    verts, lines = [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                p = ln.split(); verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith('l '):
                lines.append([int(t.split('/')[0]) - 1 for t in ln.split()[1:]])
    verts = np.array(verts)
    return [verts[c] for c in lines]


def resample_by_arc_length(points, n):
    """n points evenly spaced along the polyline's arc length."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(targets, s, points[:, k])
    return out


def foot_mlt_hours(line):
    """MLT of the line's northern foot point (its first vertex), via SM."""
    x_sm, y_sm, z_sm = geopack.smgsm(line[0, 0], line[0, 1], line[0, 2], -1)
    return (math.degrees(math.atan2(y_sm, x_sm)) / 15.0 + 12.0) % 24.0


def loft_surface(lines, name, max_gap_hours=1.01, flux_fn=None):
    """Stitch the field lines, ordered by foot MLT, into a triangle surface.
    Gaps wider than max_gap_hours (missing sectors) are left open. With
    flux_fn (points -> flux), the surface carries a 'flux' point scalar so it
    can be coloured with the same table as the cut face."""
    ordered = sorted(lines, key=foot_mlt_hours)
    mlts = [foot_mlt_hours(l) for l in ordered]
    strips = [resample_by_arc_length(l, POINTS_PER_LINE) for l in ordered]

    points = pvtk.vtkPoints()
    for strip in strips:
        for p in strip:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    tris = pvtk.vtkCellArray()
    n_lines = len(strips)
    for i in range(n_lines):
        j = (i + 1) % n_lines
        gap = (mlts[j] - mlts[i]) % 24.0
        if gap > max_gap_hours:
            continue
        for k in range(POINTS_PER_LINE - 1):
            a, b = i * POINTS_PER_LINE + k, i * POINTS_PER_LINE + k + 1
            c, d = j * POINTS_PER_LINE + k, j * POINTS_PER_LINE + k + 1
            for tri in ((a, c, b), (b, c, d)):
                tris.InsertNextCell(3)
                for pid in tri:
                    tris.InsertCellPoint(pid)
    poly = pvtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(tris)
    if flux_fn is not None:
        vals = flux_fn(np.vstack(strips))
        arr = numpy_support.numpy_to_vtk(np.asarray(vals, dtype=np.float32), deep=True)
        arr.SetName('flux')
        poly.GetPointData().SetScalars(arr)
    producer = TrivialProducer(registrationName=name)
    producer.GetClientSideObject().SetOutput(poly)
    producer.UpdatePipeline()
    return producer


def arc_in_cut_plane(lines, mlt_target):
    """The line whose foot MLT is closest to mlt_target (0 = midnight arc,
    12 = noon arc of the cut plane), as (x, z) points north foot -> south."""
    best = min(lines, key=lambda l: min(abs(foot_mlt_hours(l) - mlt_target),
                                        24.0 - abs(foot_mlt_hours(l) - mlt_target)))
    return best[:, [0, 2]]


def cross_section_mask(outer_lines, inner_lines, xz_points):
    """True for cut-plane points inside the donut: between the outer and
    inner shells' midnight arcs (nightside) or noon arcs (dayside)."""
    inside = np.zeros(len(xz_points), dtype=bool)
    for mlt in (0.0, 12.0):
        outer_arc = arc_in_cut_plane(outer_lines, mlt)
        inner_arc = arc_in_cut_plane(inner_lines, mlt)
        # closed loop: out along the outer arc, back along the inner arc
        loop = np.vstack([outer_arc, inner_arc[::-1]])
        inside |= PolygonPath(loop).contains_points(xz_points)
    return inside


# --- flux on the cut plane ----------------------------------------------------

def dipole_footpoint_lookup(points_gsm, handoff_re, wanted):
    """For points inside the handoff sphere (and flagged in `wanted`): foot
    latitude, MLT and local |B| from the analytic tilted dipole (the same
    field the tracer uses there). Returns (lat_deg, mlt_hr, b_tesla),
    NaN elsewhere and where r < 1."""
    n = len(points_gsm)
    lat_deg = np.full(n, np.nan); mlt_hr = np.full(n, np.nan); b_t = np.full(n, np.nan)
    for k in np.where(wanted)[0]:
        x, y, z = points_gsm[k]
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1.0 or r > handoff_re:
            continue
        lat_here, mlt_here, _ = dipole.gsm_point_to_sm_lat_mlt_r(x, y, z)
        l_shell = r / math.cos(math.radians(lat_here)) ** 2
        lat_deg[k] = dipole.latitude_where_line_crosses_radius(l_shell, 1.0)
        mlt_hr[k] = mlt_here
        sin_lat = math.sin(math.radians(lat_here))
        b_t[k] = EARTH_DIPOLE_TESLA / r ** 3 * math.sqrt(1.0 + 3.0 * sin_lat * sin_lat)
    return lat_deg, mlt_hr, b_t


class FluxSampler(object):
    """Band flux at arbitrary GSM points through the SAME foot-point mapping
    the volume uses: outside the handoff sphere the point's foot point comes
    from BATSRUS (theta_1 / phi_1), inside it from the analytic dipole; the RBE
    distribution at that foot point is Liouville-mapped to the local |B|."""

    def __init__(self, plt_path, npz_path, t, band, handoff_re, quantity='omni'):
        zf = np.load(npz_path, allow_pickle=True)
        E_lvls = zf['E_lvls']; self.sin_grid = np.asarray(zf['alpha_lvls'], dtype=float)
        lat_lvls = np.asarray(zf['lat_lvls'], dtype=float)
        mlt_lvls = np.asarray(zf['mlt_lvls'], dtype=float)
        bo_tesla = np.asarray(zf['bo'][t], dtype=float)
        irm_counts = np.asarray(zf['irm'][t, 0, :], dtype=float)
        zf.close()
        flux_frame = mapping.read_flux_frame(npz_path, t)
        self.rbe_grid = fpv.RbeFootpointGrid(lat_lvls, mlt_lvls, flux_frame, bo_tesla, irm_counts)
        band_ch = [np.where((E_lvls >= lo) & (E_lvls < hi))[0] for _, lo, hi in BANDS]
        self.ch = band_ch[band]
        self.widths = mapping.channel_widths(E_lvls)
        self.plt_path = plt_path
        self.handoff_re = handoff_re
        self.quantity = quantity

    def at_points(self, points):
        """Flux at each (x, y, z) GSM point; NaN where no closed foot point."""
        points = np.asarray(points, dtype=float)
        n_all = len(points)
        r = np.linalg.norm(points, axis=1)
        foot_lat = np.full(n_all, np.nan); foot_mlt = np.full(n_all, np.nan)
        b_tesla = np.full(n_all, np.nan); usable = np.zeros(n_all, dtype=bool)
        sample_idx = np.where(r >= self.handoff_re)[0]
        if len(sample_idx):
            arrays = fpv.sample_batsrus_at(self.plt_path, points[sample_idx])
            b_nt = np.sqrt(arrays['B_x_nT'] ** 2 + arrays['B_y_nT'] ** 2 + arrays['B_z_nT'] ** 2)
            foot_lat[sample_idx] = arrays['theta_1_deg']
            foot_mlt[sample_idx] = (arrays['phi_1_deg'] / 15.0 + 12.0) % 24.0
            b_tesla[sample_idx] = b_nt * 1e-9
            ok = arrays['Status'] == fpv.CLOSED_STATUS
            if 'vtkValidPointMask' in arrays:
                ok &= arrays['vtkValidPointMask'] != 0
            usable[sample_idx] = ok
        d_lat, d_mlt, d_b = dipole_footpoint_lookup(points, self.handoff_re, np.ones(n_all, dtype=bool))
        inner = np.isfinite(d_lat)
        foot_lat[inner] = d_lat[inner]; foot_mlt[inner] = d_mlt[inner]; b_tesla[inner] = d_b[inner]
        usable |= inner

        flux = np.full(n_all, np.nan)
        for k in np.where(usable)[0]:
            pad, bo = self.rbe_grid.at(foot_lat[k], foot_mlt[k])
            if pad is None:
                continue
            j_eq = (pad[self.ch] * self.widths[self.ch, None]).sum(axis=0)
            if self.quantity == 'omni':
                val = mapping.adiabatic_omni(b_tesla[k], bo, j_eq, self.sin_grid)
            else:
                val = mapping.adiabatic_perp(b_tesla[k], bo, j_eq, self.sin_grid)
            if np.isfinite(val) and val > 0:
                flux[k] = val
        return flux


def shade_cut_plane(sampler, resolution_re, extent_re, inside_fn):
    """Flux on a (x, z) grid in the y = 0 plane, computed only where
    inside_fn(xz_points) says the point is inside the donut's cross-section.
    Returns (xs, zs, image[iz, ix])."""
    xs = np.arange(-extent_re, extent_re + resolution_re / 2, resolution_re)
    zs = xs.copy()
    X, Z = np.meshgrid(xs, zs)                       # [iz, ix]
    points = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    inside = inside_fn(points[:, [0, 2]])
    print('  {} of {} grid points lie inside the cross-section'.format(inside.sum(), len(points)))
    image = np.full(X.size, np.nan)
    image[inside] = sampler.at_points(points[inside])
    return xs, zs, image.reshape(X.shape)


def cut_face_source(xs, zs, image, name):
    n_x, n_z = len(xs), len(zs)
    img = pvtk.vtkImageData()
    img.SetDimensions(n_x, 1, n_z)
    img.SetSpacing(xs[1] - xs[0], 1.0, zs[1] - zs[0])
    img.SetOrigin(xs[0], 0.0, zs[0])
    arr = numpy_support.numpy_to_vtk(image.ravel(order='C').astype(np.float32), deep=True)
    arr.SetName('flux')
    img.GetPointData().SetScalars(arr)
    producer = TrivialProducer(registrationName=name)
    producer.GetClientSideObject().SetOutput(img)
    producer.UpdatePipeline()
    return producer


# --- scene ---------------------------------------------------------------------

def show_half_surface(producer, view, color, opacity=1.0, by_flux=False):
    clip = Clip(Input=producer)
    clip.ClipType = 'Plane'
    clip.ClipType.Origin = [0.0, 0.0, 0.0]
    clip.ClipType.Normal = [0.0, 1.0, 0.0]        # keep y >= 0 (dusk half)
    clip.Invert = 0
    surface = ExtractSurface(Input=clip)           # Clip yields an unstructured grid
    normals = GenerateSurfaceNormals(Input=surface)
    normals.FeatureAngle = 70.0                    # smooth shading across the loft
    disp = Show(normals, view)
    disp.Representation = 'Surface'
    if by_flux:
        ColorBy(disp, ('POINTS', 'flux'))
        disp.Ambient = 0.45
        disp.Diffuse = 0.65
        disp.Specular = 0.15
    else:
        disp.DiffuseColor = list(color)
        disp.AmbientColor = list(color)
        disp.Ambient = 0.25
        disp.Diffuse = 0.85
        disp.Specular = 0.3
    disp.SpecularPower = 40.0
    disp.Opacity = opacity
    disp.Interpolation = 'Gouraud'
    return disp


# --- scene annotations (Sun arrow, GEI triad, caption) --------------------------

ANNOT_Z = -3.9                  # world z of the annotation row (below the belt, in front of the cut)
ANNOT_Y = -2.0                  # toward the camera so nothing occludes it


def show_arrow(view, origin, direction, length, color, label, name, label_scale=0.55):
    """A 3-D arrow (tube shaft + cone tip) from origin along direction, with a
    flat label standing in the x-z plane (facing the camera on the dawn side)."""
    v = np.asarray(direction, dtype=float); v = v / np.linalg.norm(v)
    o = np.asarray(origin, dtype=float)
    tip = o + v * length
    shaft = Line(registrationName=name + 'Shaft')
    shaft.Point1 = list(o); shaft.Point2 = list(o + v * (length - 0.28))
    d = Show(shaft, view)
    d.DiffuseColor = list(color); d.AmbientColor = list(color); d.Ambient = 0.6
    d.LineWidth = 4.0
    try:
        d.RenderLinesAsTubes = 1
    except Exception:
        pass
    cone = Cone(registrationName=name + 'Tip')
    cone.Center = list(o + v * (length - 0.14)); cone.Direction = list(v)
    cone.Height = 0.28; cone.Radius = 0.09; cone.Resolution = 24
    c = Show(cone, view)
    c.DiffuseColor = list(color); c.AmbientColor = list(color); c.Ambient = 0.6
    if label:
        txt = a3DText(registrationName=name + 'Label'); txt.Text = label
        tr = Transform(registrationName=name + 'LabelXf', Input=txt)
        tr.Transform.Rotate = [90.0, 0.0, 0.0]            # x-y text -> x-z plane, faces -y
        tr.Transform.Scale = [label_scale] * 3
        tr.Transform.Translate = list(tip + np.array([0.12, 0.0, -0.1]))
        t = Show(tr, view)
        t.DiffuseColor = list(color); t.AmbientColor = list(color); t.Ambient = 0.9; t.Diffuse = 0.2
    return tip


AXIS_COLORS = ((1.0, 0.35, 0.35), (0.4, 1.0, 0.4), (0.45, 0.6, 1.0))   # x red, y green, z blue


def show_triad(view, origin, vectors, suffix, name, length=1.1):
    """Three labelled arrows from origin along `vectors` (rows), e.g. the GSM or
    GEI unit vectors expressed in the GSM scene."""
    origin = np.asarray(origin, dtype=float)
    for k in range(3):
        show_arrow(view, origin, vectors[k], length, AXIS_COLORS[k],
                   'XYZ'[k] + ' ' + suffix, name + 'XYZ'[k], label_scale=0.18)
    dot = Sphere(registrationName=name + 'Origin'); dot.Radius = 0.07; dot.Center = list(origin)
    d = Show(dot, view); d.DiffuseColor = [1, 1, 1]; d.AmbientColor = [1, 1, 1]; d.Ambient = 0.6


def show_sun_and_axes(view):
    """A row below the belt, left to right: a yellow arrow toward the Sun
    (+x GSM), the GSM triad (the scene's own axes) and the GEI triad of this
    frame (the frame the delivered OBJs are in), all drawn in GSM."""
    show_arrow(view, [-9.6, ANNOT_Y, ANNOT_Z + 0.9], [1.0, 0.0, 0.0], 1.8, (1.0, 0.85, 0.2),
               'to the Sun', 'Sun', label_scale=0.22)
    show_triad(view, [-5.2, ANNOT_Y, ANNOT_Z + 0.4], np.eye(3), 'GSM', 'Gsm')
    M = frames.rotation_matrix(frames.gsm_to_gei)      # M @ v_gsm = v_gei, so rows = GEI axes in GSM
    show_triad(view, [-1.2, ANNOT_Y, ANNOT_Z], M, 'GEI', 'Gei')


def caption_text(ts, band_label, by_flux, flux_min=1e3, flux_max=FLUX_MAX):
    lines = [
        'FRAME     {}   {}-{}-{} {}:{} UT'.format(ts, ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13]),
        'AXES      GSM (+x toward the Sun, +z north).  Camera fixed on the dawn side (y < 0).',
        'CUT       y = 0, the noon-midnight meridian (same plane in GSM and SM); dusk half (y >= 0) drawn.',
        'SURFACES  obj_fp_outer + obj_fp_inner fieldlines_<frame>.obj lofted between neighbouring lines;',
        '          delivered obj_fp_*_gei = the same lines rotated to GEI (gsm_transforms_fp.csv).',
        'COLOUR    {} omnidirectional electron flux, log {:g} - {:g} cm^-2 s^-1, {}.'.format(
            band_label, flux_min, flux_max, 'cut face and both shells' if by_flux else 'cut face only'),
    ]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True)
    p.add_argument('--band', type=int, default=2)
    p.add_argument('--resolution', type=float, default=0.025, help='cut-plane grid (Re)')
    p.add_argument('--extent', type=float, default=10.0)
    p.add_argument('--flux-min', type=float, default=1e3)
    p.add_argument('--flux-max', type=float, default=FLUX_MAX)
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--out', default=None)
    p.add_argument('--size', default='1920x1080')
    p.add_argument('--shell-color', choices=['flux', 'solid'], default='flux',
                   help='flux: shells carry the same flux colour table as the cut '
                        'face (edge colours match); solid: flat orange')
    p.add_argument('--no-caption', action='store_true')
    args = p.parse_args()
    ts = args.ts
    out_png = args.out or os.path.join(args.out_root, 'fp_cutaway_{}.png'.format(ts))
    width, height = [int(v) for v in args.size.split('x')]

    cfg = config.CRAConfig()
    frame_dt = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
    frames.recalc_for(frame_dt)                    # SM <-> GSM for this instant
    t = data.frame_index_for(cfg.paths.npz_path, frame_dt, data.base_datetime(cfg.time.base_iso))
    if t is None:
        sys.exit('[{}] no flux frame at this time in {}'.format(ts, cfg.paths.npz_path))
    plt_path = render.find_matching_plt(cfg.paths.plt_dir, ts)

    outer_lines = read_obj_polylines(os.path.join(args.out_root, 'obj_fp_outer',
                                                  'fieldlines_{}.obj'.format(ts)))
    inner_lines = read_obj_polylines(os.path.join(args.out_root, 'obj_fp_inner',
                                                  'fieldlines_{}.obj'.format(ts)))

    sampler = FluxSampler(plt_path, cfg.paths.npz_path, t, args.band, dipole.HANDOFF_RADIUS_RE)
    print('[{}] shading the cut plane at {:.3f} Re resolution ...'.format(ts, args.resolution))
    xs, zs, image = shade_cut_plane(
        sampler, args.resolution, args.extent,
        inside_fn=lambda xz: cross_section_mask(outer_lines, inner_lines, xz))
    image[np.isfinite(image) & (image < args.flux_min)] = args.flux_min   # floor, not hole
    print('[{}] {} shaded points inside the cross-section'.format(ts, np.isfinite(image).sum()))

    def shell_flux(points):
        """Flux on the lofted shells, floored like the cut face (no holes).
        Vertices whose foot point lands on an empty grid cell (NaN or zero --
        CIMI's inner edge is sharp and BATSRUS's foot latitude sits ~1 deg
        equatorward of the grid's) take the value of the nearest mapped vertex
        instead of painting a black streak."""
        vals = sampler.at_points(points)
        bad = ~np.isfinite(vals) | (vals <= 0.0)
        if bad.any() and (~bad).any():
            good_pts, good_vals = points[~bad], vals[~bad]
            for i in np.where(bad)[0]:
                d2 = ((good_pts - points[i]) ** 2).sum(axis=1)
                vals[i] = good_vals[int(d2.argmin())]
        vals[~np.isfinite(vals) | (vals < args.flux_min)] = args.flux_min
        return vals
    flux_fn = shell_flux if args.shell_color == 'flux' else None

    view = CreateView('RenderView')
    view.ViewSize = [width, height]
    view.Background = [0.0, 0.0, 0.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0

    # the donut: outer skin + hole wall (flux-coloured unless --shell-color solid)
    by_flux = flux_fn is not None
    show_half_surface(loft_surface(outer_lines, 'OuterSurface', flux_fn=flux_fn), view,
                      (0.95, 0.55, 0.12), by_flux=by_flux)
    show_half_surface(loft_surface(inner_lines, 'InnerSurface', flux_fn=flux_fn), view,
                      (0.85, 0.45, 0.10), by_flux=by_flux)

    # the cut face
    face = cut_face_source(xs, zs, image, 'CutFace')
    face_disp = Show(face, view)
    face_disp.Representation = 'Surface'
    ColorBy(face_disp, ('POINTS', 'flux'))
    lut = GetColorTransferFunction('flux')
    lut.ApplyPreset('Inferno (matplotlib)', True)
    lut.RescaleTransferFunction(args.flux_min, args.flux_max)
    lut.MapControlPointsToLogSpace()
    lut.UseLogScale = 1
    lut.NanOpacity = 0.0
    face_disp.Ambient = 0.7
    face_disp.Diffuse = 0.3
    face_disp.SetScalarBarVisibility(view, True)
    sbar = GetScalarBar(lut, view)
    sbar.Title = 'omnidirectional flux, {}'.format(BAND_LABELS[args.band])
    sbar.ComponentTitle = '(cm^-2 s^-1)'
    sbar.TitleColor = [1, 1, 1]; sbar.LabelColor = [1, 1, 1]
    sbar.WindowLocation = 'Lower Right Corner'
    sbar.ScalarBarLength = 0.35

    earth = Sphere(registrationName='Earth')
    earth.Radius = 1.0; earth.ThetaResolution = 96; earth.PhiResolution = 96
    earth_disp = Show(earth, view)
    earth_disp.DiffuseColor = [0.23, 0.43, 0.65]
    earth_disp.AmbientColor = [0.23, 0.43, 0.65]

    label = Text(registrationName='Title')
    label.Text = 'Radiation belt cut at the noon-midnight meridian   {}-{}-{} {}:{} UT'.format(
        ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13])
    tdisp = Show(label, view)
    tdisp.WindowLocation = 'Upper Center'; tdisp.FontSize = 22; tdisp.Color = [1, 1, 1]

    if not args.no_caption:
        caption = Text(registrationName='Caption')
        caption.Text = caption_text(ts, BAND_LABELS[args.band], by_flux, args.flux_min, args.flux_max)
        cdisp = Show(caption, view)
        cdisp.WindowLocation = 'Lower Left Corner'; cdisp.FontSize = 14
        cdisp.Color = [0.92, 0.92, 0.92]; cdisp.Justification = 'Left'
        for prop, val in (('FontFamily', 'Courier'), ('BackgroundColor', [0.08, 0.08, 0.1]),
                          ('BackgroundOpacity', 0.75), ('ShowBackground', 1), ('Padding', 6)):
            try:
                setattr(cdisp, prop, val)
            except Exception:
                pass
        show_sun_and_axes(view)

    view.CameraPosition = [-5.0, -18.0, 6.0]
    view.CameraFocalPoint = [-0.5, 0.0, -0.9]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraViewAngle = 33.0
    Render(view)
    SaveScreenshot(out_png, view, ImageResolution=[width, height])
    print('saved', out_png)


if __name__ == '__main__':
    main()
