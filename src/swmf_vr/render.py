"""Tracing the field lines and the preview screenshot -- the only part that
needs ParaView. Import this under pvbatch, not the system python3.

Trace the streamlines, pull plain points/cells/scalars out of the VTK result
(color.write_colored_obj takes it from there), and render a quick colored PNG.
"""

import os
import re
import glob
import math

from paraview.simple import (
    CSVReader, TableToPoints, StreamTracerWithCustomSource,
    VisItTecplotBinaryReader, Calculator, CreateView, Delete,
    TrivialProducer, Show, ColorBy, GetColorTransferFunction,
    _DisableFirstRenderCameraReset,
)
from paraview import servermanager as _sm

from .config import SEED_SCALARS, CMAP_PRESETS, resolve_color_range

_DisableFirstRenderCameraReset()


def make_view(camera):
    view = CreateView('RenderView')
    view.ViewSize = list(camera.view_size)
    view.InteractionMode = '3D'
    view.CameraPosition = list(camera.position)
    view.CameraFocalPoint = list(camera.focal_point)
    view.CameraViewUp = list(camera.view_up)
    view.Background = list(camera.background)
    view.UseColorPaletteForBackground = 0
    return view


def find_matching_plt(plt_dir, ts):
    parts = ts.split('_')
    if len(parts) != 2:
        return None
    matches = glob.glob(os.path.join(plt_dir, "*e{}-{}*.plt".format(parts[0], parts[1])))
    return matches[0] if matches else None


def collect_timestamps(seed_dirs):
    stamps = set()
    for d in seed_dirs:
        for p in glob.glob(os.path.join(d, 'boundary_*.csv')):
            m = re.match(r'boundary_(\d{8}_\d{6})\.csv', os.path.basename(p))
            if m:
                stamps.add(m.group(1))
    return sorted(stamps)


def load_b_field(plt_path):
    """Read a .plt and build the 'B' vector. Returns (reader, b_calc); Delete both."""
    reader = VisItTecplotBinaryReader(registrationName='GM_Reader', FileName=[plt_path])
    reader.MeshStatus = ['global_field']
    reader.PointArrayStatus = ['B_x_nT', 'B_y_nT', 'B_z_nT', 'x', 'y', 'z']
    reader.UpdatePipeline()
    b_calc = Calculator(registrationName='B_Vector', Input=reader)
    b_calc.ResultArrayName = 'B'
    b_calc.Function = 'B_x_nT*iHat+B_y_nT*jHat+B_z_nT*kHat'
    return reader, b_calc


def trace_streamlines(b_calc, csv_path, trace_cfg):
    """Trace from one seed CSV. Returns (tracer, points_source, csv_reader, poly,
    seed_poly) or None. Proxies stay alive for the caller to Delete."""
    csv_reader = points_source = tracer = None
    try:
        csv_reader = CSVReader(registrationName='Seed_Reader', FileName=[csv_path])
        csv_reader.HaveHeaders = 1
        points_source = TableToPoints(registrationName='Seed_Points', Input=csv_reader)
        points_source.XColumn = 'Seed_X_Re'
        points_source.YColumn = 'Seed_Y_Re'
        points_source.ZColumn = 'Seed_Z_Re'
        points_source.KeepAllDataArrays = 1
        tracer = StreamTracerWithCustomSource(registrationName='Tracer',
                                              Input=b_calc, SeedSource=points_source)
        tracer.Vectors = list(trace_cfg.vectors)
        tracer.MaximumStreamlineLength = trace_cfg.max_streamline_length
        tracer.IntegrationDirection = trace_cfg.integration_direction
        tracer.UpdatePipeline()
        return (tracer, points_source, csv_reader,
                _sm.Fetch(tracer), _sm.Fetch(points_source))
    except Exception as e:
        print("  trace failed ({})".format(e))
        for proxy in (tracer, points_source, csv_reader):
            if proxy is not None:
                Delete(proxy)
        return None


def extract_geometry_and_scalar(poly, seed_poly, scalar):
    """Pull plain (points, cells, point_vals) out of the traced VTK result.
    Per-line scalars come from the seed via the SeedIds cell array; along-line
    ones are computed per vertex."""
    n = poly.GetNumberOfPoints()
    points = [poly.GetPoint(i) for i in range(n)]
    cells = []
    for c in range(poly.GetNumberOfCells()):
        ids = poly.GetCell(c).GetPointIds()
        cells.append([ids.GetId(k) for k in range(ids.GetNumberOfIds())])

    vals = [0.0] * n
    if scalar in SEED_SCALARS:
        seed_arr = seed_poly.GetPointData().GetArray(SEED_SCALARS[scalar])
        if seed_arr is None:
            raise ValueError("seed column '{}' missing".format(SEED_SCALARS[scalar]))
        seed_vals = [seed_arr.GetValue(i) for i in range(seed_arr.GetNumberOfTuples())]
        seedids = poly.GetCellData().GetArray('SeedIds')
        for ci, cell in enumerate(cells):
            sv = seed_vals[int(seedids.GetValue(ci))]
            for pid in cell:
                vals[pid] = sv
    elif scalar == 'bmag':
        B = poly.GetPointData().GetArray('B')
        for i in range(n):
            bx, by, bz = B.GetTuple3(i)
            vals[i] = math.sqrt(bx * bx + by * by + bz * bz)
    elif scalar == 'radius':
        for i in range(n):
            x, y, z = points[i]
            vals[i] = math.sqrt(x * x + y * y + z * z)
    elif scalar == 'lat':
        for i in range(n):
            x, y, z = points[i]
            r = math.sqrt(x * x + y * y + z * z)
            vals[i] = math.degrees(math.asin(z / r)) if r > 0 else 0.0
    else:
        raise ValueError("unknown scalar '{}'".format(scalar))
    return points, cells, vals


def show_scalar_colored(view, poly, point_vals, shading):
    """Color the lines by point_vals using the same colormap/range the OBJ uses.
    Returns the producer proxy (Delete it) or None to fall back to a flat color."""
    try:
        from paraview import vtk as pvtk
        arr = pvtk.vtkFloatArray()
        arr.SetName('ColorScalar')
        arr.SetNumberOfTuples(len(point_vals))
        for i, v in enumerate(point_vals):
            arr.SetValue(i, float(v))
        poly.GetPointData().AddArray(arr)

        tp = TrivialProducer(registrationName='ColoredLines')
        tp.GetClientSideObject().SetOutput(poly)
        tp.UpdatePipeline()
        disp = Show(tp, view)
        disp.Representation = 'Surface'
        ColorBy(disp, ('POINTS', 'ColorScalar'))
        log, vmin, vmax = resolve_color_range(shading)
        ctf = GetColorTransferFunction('ColorScalar')
        preset = CMAP_PRESETS.get(shading.cmap)
        if preset:
            ctf.ApplyPreset(preset, True)
        ctf.RescaleTransferFunction(vmin, vmax)
        if log:
            ctf.MapControlPointsToLogSpace()
            ctf.UseLogScale = 1
        disp.SetScalarBarVisibility(view, False)
        return tp
    except Exception as e:
        print("    preview coloring failed ({})".format(e))
        return None
