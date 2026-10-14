#!/usr/bin/env python3
"""Foot-point seed CSVs -> field-line OBJs built the hybrid way (pvbatch).

  "$PV_BATCH" scripts/footpoint_trace.py [rank size [max_files]] [--match SUBSTR]
       [--color-by NAME] [--cmap NAME] [--vmin V] [--vmax V] [--log/--no-log]
       [--out-root DIR] [--plt-dir DIR] [--handoff RE] [--apex-max RE]
       [--skip-existing]

How each field line is built
----------------------------
Every boundary foot point from footpoint_seeds.py becomes one polyline with
up to three pieces, joined at the handoff sphere (2.5 Re -- see
swmf_vr/dipole.py for why):

  1. NORTH ANALYTIC PIECE: the tilted centered dipole line from the northern
     foot point (r = 1 Re) up to the handoff sphere. Computed in SM
     coordinates where the dipole is a textbook formula, then rotated to GSM.

  2. NUMERIC PIECE: from the handoff crossing, the line is integrated
     through the actual BATSRUS field (ParaView stream tracer on the .plt
     B field). We integrate BOTH directions from the seed and keep whichever
     side actually goes out into the magnetosphere; the other side dives
     straight into the body and is discarded. The kept side is CUT at the
     first re-entry into the handoff sphere -- the trace never uses the
     non-physical field values inside rBody (the old pipeline's defect).

  3. SOUTH ANALYTIC PIECE: from wherever the numeric piece re-entered the
     handoff sphere, the dipole line through that point is followed down to
     the Earth's surface, in the hemisphere the trace actually ended in.
     (No mirror-symmetry assumption on this end -- the numeric trace tells
     us where the line really came down to 2.5 Re.)

Low lines whose whole extent is inside the handoff sphere (L <= 2.5, marked
Trace_MHD = 0 in the CSV) are drawn fully analytically, north foot over the
apex to the mirror south foot.

The OBJ output format, coloring, and directory layout mirror the old
trace.py (obj_fp_* instead of obj_*), so the downstream GEI conversion and
the VR app need no changes.
"""

import os
import sys
import math
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# pvbatch's bundled python does not have geopack installed; the system python
# does (pip install --user), and geopack is pure python, so borrowing it from
# there is safe. APPEND (never prepend) so pvbatch keeps its own numpy/scipy.
try:
    from geopack import geopack as _geopack_probe  # noqa: F401
except (ModuleNotFoundError, ValueError):
    import glob
    for _site_dir in sorted(glob.glob(
            os.path.expanduser('~/.local/lib/python*/site-packages'))):
        if not os.path.isdir(os.path.join(_site_dir, 'geopack')):
            continue
        sys.path.append(_site_dir)
        try:
            from geopack import geopack as _geopack_probe  # noqa: F401
            break
        except Exception:
            # This copy is broken under this python -- try the next one.
            sys.path.remove(_site_dir)
            for _mod in [m for m in sys.modules if m.startswith('geopack')]:
                del sys.modules[_mod]

from paraview.simple import (
    StreamTracerWithCustomSource, TrivialProducer, Delete,
    _DisableFirstRenderCameraReset,
)
from paraview import servermanager as _sm
from paraview import vtk as pvtk

from swmf_vr import config, render, dipole, frames
from swmf_vr.color import write_colored_obj, write_color_assets
from geopack import geopack

_DisableFirstRenderCameraReset()

# Points on the numeric trace this close below the handoff radius still count
# as "on the sphere" -- the integrator does not land exactly on it.
HANDOFF_TOLERANCE_RE = 0.02

# A boundary line that traces OPEN in the instantaneous BATSRUS field cannot
# hold trapped particles, whatever RBE's irm said (irm lags: it comes from
# RBE's own coupled-field updates, and during sudden compressions -- e.g. the
# 17:5x SSC frames -- the outermost "closed" foot points have just opened).
# So the tracer retreats the foot point equatorward in small latitude steps
# until the line closes, and drops it if it still hasn't after MAX_RETREATS.
RETREAT_STEP_DEG = 0.5
MAX_RETREATS = 12          # up to 6 degrees of retreat

# The same retreat is applied when a CLOSED line reaches beyond this radius.
# RBE sometimes carries above-threshold flux on foot points whose BATSRUS
# line stretches to 15-40 Re during the storm (e.g. 06:51 May 11, MLT 3.5:
# flux ~1000 at ro ~ 10 Re). MeV electrons cannot stay trapped on such lines
# (magnetopause shadowing / tail loss within a drift), and in the VR product
# they appear as giant bumps on the belt surface. 12 Re is the limit the old
# volume mapping used for "closed"; override with --apex-max. MODELING CHOICE.
APEX_MAX_RE = 12.0

# Which CSV column colors a whole line, per --color-by name.
FP_SEED_SCALARS = {'eqflux': 'Flux', 'eqflux_peak': 'Flux_peak',
                   'lshell': 'L_shell'}
FP_POINT_SCALARS = ('radius', 'lat')


def parse_args(argv):
    rank, size, max_files, match = 0, 1, None, None
    over, out_root, plt_dir = {}, None, None
    handoff = dipole.HANDOFF_RADIUS_RE
    skip_existing = False
    apex_max = APEX_MAX_RE
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
        if a == '--handoff': handoff = float(argv[i + 1]); i += 2; continue
        if a == '--skip-existing': skip_existing = True; i += 1; continue
        if a == '--apex-max': apex_max = float(argv[i + 1]); i += 2; continue
        i += 1
    return (rank, size, max_files, match, over, out_root, plt_dir, handoff,
            skip_existing, apex_max)


def read_footpoint_seed_csv(path):
    """Rows of the foot-point seed CSV as dictionaries keyed by column name."""
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) <= 1:
        return []
    cols = lines[0].split(',')
    rows = []
    for ln in lines[1:]:
        vals = [float(v) for v in ln.split(',')]
        rows.append(dict(zip(cols, vals)))
    return rows


def make_seed_source(seed_rows):
    """A ParaView point source holding the handoff seed points of the rows
    that need a numeric trace. Returns the producer proxy (Delete it)."""
    points = pvtk.vtkPoints()
    verts = pvtk.vtkCellArray()
    for row in seed_rows:
        pid = points.InsertNextPoint(row['Seed_X_Re'], row['Seed_Y_Re'],
                                     row['Seed_Z_Re'])
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
    poly = pvtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)
    producer = TrivialProducer(registrationName='FP_Seeds')
    producer.GetClientSideObject().SetOutput(poly)
    producer.UpdatePipeline()
    return producer


def radius_of(point):
    x, y, z = point
    return math.sqrt(x * x + y * y + z * z)


def interpolate_sphere_crossing(point_outside, point_inside, sphere_radius_re):
    """The point on the segment between the two points where r equals the
    sphere radius (linear interpolation in r along the segment)."""
    r_out = radius_of(point_outside)
    r_in = radius_of(point_inside)
    if r_in == r_out:
        return point_outside
    fraction = (sphere_radius_re - r_out) / (r_in - r_out)
    fraction = min(1.0, max(0.0, fraction))
    return tuple(point_outside[k] + fraction * (point_inside[k] - point_outside[k])
                 for k in range(3))


def walk_from(cell_points, start_index, step, handoff_re):
    """Walk a traced polyline from start_index in one direction (step = +1
    or -1), keeping points while they stay on or outside the handoff sphere.
    Stops (and appends the exact sphere crossing) at the first point inside.

    Returns (kept_points, reentered): reentered is False when the walk ran
    off the end of the polyline without coming back inside -- an open or
    escaping line."""
    kept = []
    i = start_index
    while 0 <= i < len(cell_points):
        point = cell_points[i]
        if radius_of(point) >= handoff_re - HANDOFF_TOLERANCE_RE:
            kept.append(point)
            i += step
        else:
            if kept:
                kept.append(interpolate_sphere_crossing(kept[-1], point, handoff_re))
            return kept, True
    return kept, False


def extract_magnetospheric_arc(seed_point, cells_points, handoff_re):
    """From the stream-trace output for one seed (a list of polylines, each
    a list of (x,y,z)), recover the piece that actually goes out into the
    magnetosphere.

    The tracer integrated BOTH directions from the seed: one side dives
    below the handoff sphere immediately (the earthward stub), the other
    goes around through the magnetosphere. We walk away from the seed in
    every available direction and keep the longest walk -- that is the arc.

    Returns (arc_points, reentered) or (None, False) if nothing usable."""
    best_arc, best_reentered, best_length = None, False, -1.0
    for cell_points in cells_points:
        if len(cell_points) < 2:
            continue
        # The point on this polyline closest to the seed (with direction
        # BOTH the seed sits in the middle of the merged line).
        distances = [sum((cell_points[i][k] - seed_point[k]) ** 2
                         for k in range(3))
                     for i in range(len(cell_points))]
        i_seed = distances.index(min(distances))
        for step in (1, -1):
            arc, reentered = walk_from(cell_points, i_seed, step, handoff_re)
            if len(arc) < 2:
                continue
            length = sum(math.dist(arc[i], arc[i + 1])
                         for i in range(len(arc) - 1))
            if length > best_length:
                best_arc, best_reentered, best_length = arc, reentered, length
    return best_arc, best_reentered


def handoff_seed_gsm(foot_lat_deg, mlt_hr, handoff_re):
    """GSM position where the dipole line from this foot point crosses the
    handoff sphere (geopack.recalc must be set for the frame)."""
    l_shell = dipole.l_shell_from_foot_latitude(foot_lat_deg)
    lat_at_handoff_deg = dipole.latitude_where_line_crosses_radius(
        l_shell, handoff_re)
    x_sm, y_sm, z_sm = dipole.sm_point_on_line(l_shell, mlt_hr,
                                               lat_at_handoff_deg)
    return geopack.smgsm(x_sm, y_sm, z_sm, 1)


def trace_one_seed(b_calc, trace_cfg, seed_point, handoff_re):
    """Stream-trace a single seed and extract its magnetospheric arc.
    Returns (arc, reentered) -- (None, False) if nothing usable."""
    seed_source = tracer = None
    try:
        seed_source = make_seed_source([{'Seed_X_Re': seed_point[0],
                                         'Seed_Y_Re': seed_point[1],
                                         'Seed_Z_Re': seed_point[2]}])
        tracer = StreamTracerWithCustomSource(
            registrationName='FP_Retry_Tracer', Input=b_calc,
            SeedSource=seed_source)
        tracer.Vectors = list(trace_cfg.vectors)
        tracer.MaximumStreamlineLength = trace_cfg.max_streamline_length
        tracer.IntegrationDirection = 'BOTH'
        tracer.UpdatePipeline()
        poly = _sm.Fetch(tracer)
        cells_points = []
        for c in range(poly.GetNumberOfCells()):
            ids = poly.GetCell(c).GetPointIds()
            cells_points.append([poly.GetPoint(ids.GetId(k))
                                 for k in range(ids.GetNumberOfIds())])
        return extract_magnetospheric_arc(seed_point, cells_points, handoff_re)
    finally:
        for proxy in (tracer, seed_source):
            if proxy is not None:
                Delete(proxy)


def line_scalar_value(row, scalar):
    return row[FP_SEED_SCALARS[scalar]]


def point_scalar_values(points, scalar):
    vals = []
    for (x, y, z) in points:
        r = math.sqrt(x * x + y * y + z * z)
        if scalar == 'radius':
            vals.append(r)
        else:                          # 'lat'
            vals.append(math.degrees(math.asin(z / r)) if r > 0 else 0.0)
    return vals


def arc_apex_re(arc):
    return max(radius_of(p) for p in arc)


def build_surface_lines(seed_rows, b_calc, trace_cfg, handoff_re, ts,
                        apex_max_re=APEX_MAX_RE):
    """All the field lines of one surface for one frame, as
    (list of polylines, list of per-line seed rows, open-line count).
    Each polyline is a list of GSM (x,y,z) points from north foot to
    south foot."""
    lines, line_rows, open_count = [], [], 0

    dipole_only_rows = [r for r in seed_rows if r['Trace_MHD'] == 0]
    traceable_rows = [r for r in seed_rows if r['Trace_MHD'] == 1]

    # 1) Low lines: fully analytic, no BATSRUS involvement.
    for row in dipole_only_rows:
        lines.append(dipole.full_dipole_line_gsm(row['Foot_Lat_deg'],
                                                 row['Foot_MLT_hr']))
        line_rows.append(row)

    # 2) Lines that reach the BATSRUS region: trace them all in one go.
    if traceable_rows:
        seed_source = tracer = None
        try:
            seed_source = make_seed_source(traceable_rows)
            tracer = StreamTracerWithCustomSource(
                registrationName='FP_Tracer', Input=b_calc,
                SeedSource=seed_source)
            tracer.Vectors = list(trace_cfg.vectors)
            tracer.MaximumStreamlineLength = trace_cfg.max_streamline_length
            tracer.IntegrationDirection = 'BOTH'
            tracer.UpdatePipeline()
            poly = _sm.Fetch(tracer)

            # Group the traced polylines by which seed they came from.
            cells_by_seed = {}
            seed_ids = poly.GetCellData().GetArray('SeedIds')
            for c in range(poly.GetNumberOfCells()):
                ids = poly.GetCell(c).GetPointIds()
                cell_points = [poly.GetPoint(ids.GetId(k))
                               for k in range(ids.GetNumberOfIds())]
                sid = int(seed_ids.GetValue(c))
                cells_by_seed.setdefault(sid, []).append(cell_points)

            for sid, row in enumerate(traceable_rows):
                seed_point = (row['Seed_X_Re'], row['Seed_Y_Re'], row['Seed_Z_Re'])
                arc, reentered = extract_magnetospheric_arc(
                    seed_point, cells_by_seed.get(sid, []), handoff_re)
                if arc is None:
                    print('[{}] seed at MLT {:.1f}: no usable trace -- skipped'
                          .format(ts, row['Foot_MLT_hr']))
                    continue

                # An open line cannot hold trapped particles, and neither can
                # one stretching beyond the trapping region: retreat the foot
                # point equatorward until the BATSRUS line is closed AND its
                # apex is within apex_max_re.
                foot_lat_deg = row['Foot_Lat_deg']
                mlt_hr = row['Foot_MLT_hr']
                went_dipolar = False

                def line_is_acceptable(arc, reentered):
                    return (arc is not None and reentered
                            and arc_apex_re(arc) <= apex_max_re)

                if not line_is_acceptable(arc, reentered):
                    reason = ('open' if not reentered
                              else 'apex {:.1f} Re'.format(arc_apex_re(arc)))
                    for attempt in range(MAX_RETREATS):
                        foot_lat_deg -= RETREAT_STEP_DEG
                        l_shell = dipole.l_shell_from_foot_latitude(foot_lat_deg)
                        if l_shell <= handoff_re:
                            went_dipolar = True   # retreated below the handoff
                            break
                        seed_point = handoff_seed_gsm(foot_lat_deg, mlt_hr,
                                                      handoff_re)
                        arc, reentered = trace_one_seed(
                            b_calc, trace_cfg, seed_point, handoff_re)
                        if line_is_acceptable(arc, reentered):
                            break
                    if line_is_acceptable(arc, reentered) or went_dipolar:
                        print('[{}] MLT {:4.1f}: {} at lat {:.2f} -> accepted at '
                              '{:.2f} ({} steps)'.format(
                                  ts, mlt_hr, reason, row['Foot_Lat_deg'],
                                  foot_lat_deg, attempt + 1))
                    else:
                        open_count += 1
                        print('[{}] MLT {:4.1f}: {} and still unacceptable after '
                              '{:.1f} deg of retreat -- line dropped'.format(
                                  ts, mlt_hr, reason,
                                  MAX_RETREATS * RETREAT_STEP_DEG))
                        continue

                if went_dipolar:
                    full_line = dipole.full_dipole_line_gsm(foot_lat_deg, mlt_hr)
                else:
                    north_piece = dipole.north_segment_gsm(
                        foot_lat_deg, mlt_hr, handoff_re)
                    south_piece = dipole.south_segment_gsm_from_reentry(*arc[-1])
                    full_line = north_piece + arc + south_piece
                lines.append(full_line)
                line_rows.append(row)
        finally:
            for proxy in (tracer, seed_source):
                if proxy is not None:
                    Delete(proxy)

    return lines, line_rows, open_count


def one_frame(cfg, ts, seed_dirs, obj_dirs, scalar, log, vmin, vmax, handoff_re,
              apex_max_re=APEX_MAX_RE):
    plt_path = render.find_matching_plt(cfg.paths.plt_dir, ts)
    if not plt_path:
        print('[{}] no .plt -- skipping'.format(ts))
        return
    if os.path.getsize(plt_path) == 0:
        # e.g. 20240511_045900: a truncated 0-byte restart-iteration file.
        # Treat it as a data gap rather than drawing an inner-only frame.
        print('[{}] empty .plt -- skipping'.format(ts))
        return

    # The SM <-> GSM rotation (dipole tilt) is frame-time dependent; set it
    # before any analytic dipole geometry is computed.
    frames.recalc_for(datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S'))

    reader = b_calc = None
    try:
        reader, b_calc = render.load_b_field(plt_path)
        for s in cfg.surfaces:
            csv_path = os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts))
            if not os.path.exists(csv_path):
                continue
            seed_rows = read_footpoint_seed_csv(csv_path)
            if not seed_rows:
                continue

            lines, line_rows, open_count = build_surface_lines(
                seed_rows, b_calc, cfg.trace, handoff_re, ts, apex_max_re)
            if not lines:
                continue

            # Flatten the polylines into the (points, cells, values) form the
            # OBJ writer takes.
            points, cells, vals = [], [], []
            for line, row in zip(lines, line_rows):
                first_index = len(points)
                points.extend(line)
                cells.append(list(range(first_index, first_index + len(line))))
                if scalar in FP_SEED_SCALARS:
                    vals.extend([line_scalar_value(row, scalar)] * len(line))
                else:
                    vals.extend(point_scalar_values(line, scalar))

            out_path = os.path.join(obj_dirs[s.name], 'fieldlines_{}.obj'.format(ts))
            umin, umax = write_colored_obj(points, cells, vals, out_path,
                                           scalar, log, vmin, vmax)
            write_color_assets(obj_dirs[s.name], cfg.shading.cmap, scalar,
                               log, vmin, vmax)

            n_dipole_only = sum(1 for r in line_rows if r['Trace_MHD'] == 0)
            r_min = min(radius_of(p) for p in points)
            print('[{}] {}: {} lines ({} fully-dipole, {} dropped-open), '
                  'r_min {:.3f} Re, U {:.3f}-{:.3f}'.format(
                      ts, s.name, len(lines), n_dipole_only, open_count,
                      r_min, umin, umax))
    except Exception as e:
        print('[{}] failed ({})'.format(ts, e))
    finally:
        if b_calc is not None:
            Delete(b_calc)
        if reader is not None:
            Delete(reader)


def frame_is_done(ts, seed_dirs, obj_dirs, surfaces):
    """A frame is complete when every surface that has seed rows also has
    its OBJ on disk (used by --skip-existing so an array job can resume)."""
    for s in surfaces:
        csv_path = os.path.join(seed_dirs[s.name], 'boundary_{}.csv'.format(ts))
        if not os.path.exists(csv_path):
            continue
        if not read_footpoint_seed_csv(csv_path):
            continue
        if not os.path.exists(os.path.join(obj_dirs[s.name],
                                           'fieldlines_{}.obj'.format(ts))):
            return False
    return True


def main():
    (rank, size, max_files, match, over, out_root, plt_dir, handoff,
     skip_existing, apex_max) = parse_args(sys.argv[1:])
    cfg = config.CRAConfig()
    if over:
        cfg = cfg._replace(shading=cfg.shading._replace(**over))
    if out_root:
        cfg = cfg._replace(paths=cfg.paths._replace(out_root=os.path.abspath(out_root)))
    if plt_dir:
        cfg = cfg._replace(paths=cfg.paths._replace(plt_dir=os.path.abspath(plt_dir)))

    scalar = cfg.shading.scalar
    if scalar not in (set(FP_SEED_SCALARS) | set(FP_POINT_SCALARS)):
        print("unknown --color-by '{}' (have: {})".format(
            scalar, sorted(set(FP_SEED_SCALARS) | set(FP_POINT_SCALARS))))
        sys.exit(1)
    log, vmin, vmax = config.resolve_color_range(cfg.shading)
    print('color: {} {} [{:g},{:g}] {}'.format(
        scalar, cfg.shading.cmap, vmin, vmax, 'log' if log else 'linear'))
    print('handoff radius: {:.3f} Re; apex cap: {:.1f} Re'.format(handoff, apex_max))

    out = cfg.paths.out_root
    seed_dirs = {s.name: os.path.join(out, 'out_seeds_fp_' + s.name)
                 for s in cfg.surfaces}
    obj_dirs = {s.name: os.path.join(out, 'obj_fp_' + s.name)
                for s in cfg.surfaces}
    for d in obj_dirs.values():
        os.makedirs(d, exist_ok=True)

    stamps = render.collect_timestamps(seed_dirs.values())
    if match is not None:
        stamps = [t for t in stamps if match in t]
    if not stamps:
        print('no foot-point seed CSVs; run footpoint_seeds.py first')
        return
    mine = stamps[rank::size]
    if max_files is not None:
        mine = mine[:max_files]
    print('{} stamps total; worker {}/{} -> {}'.format(
        len(stamps), rank, size, len(mine)))

    for ts in mine:
        if skip_existing and frame_is_done(ts, seed_dirs, obj_dirs, cfg.surfaces):
            continue
        one_frame(cfg, ts, seed_dirs, obj_dirs, scalar, log, vmin, vmax, handoff,
                  apex_max)
    print('worker {}: done'.format(rank))


if __name__ == '__main__':
    main()
