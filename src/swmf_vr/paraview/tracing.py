"""CAP 5 -- drawing concentric shells (field-line tracing). pvbatch ONLY.

Traces field lines from per-surface equatorial seeds through the GM magnetic
field (.plt) with a ParaView StreamTracer, yielding streamline polydata that the
shading layer turns into colored OBJ. Verbatim port of the v6
render_field_lines_v6 tracing pieces.

Importing this module requires ParaView (`import paraview.simple`); it runs only
under pvbatch.
"""

import os
import re
import glob

from paraview.simple import (
    CSVReader, TableToPoints, StreamTracerWithCustomSource,
    VisItTecplotBinaryReader, Calculator, CreateView, Delete,
    _DisableFirstRenderCameraReset,
)
from paraview import servermanager as _sm

_DisableFirstRenderCameraReset()


def make_render_view(camera):
    """Create the shared offscreen render view from a CameraConfig."""
    view = CreateView('RenderView')
    view.ViewSize = list(camera.view_size)
    view.InteractionMode = '3D'
    view.CameraPosition = list(camera.position)
    view.CameraFocalPoint = list(camera.focal_point)
    view.CameraViewUp = list(camera.view_up)
    view.Background = list(camera.background)
    view.UseColorPaletteForBackground = 0   # keep background fixed after ColorBy
    return view


def find_matching_plt(plt_dir, timestamp_str):
    """Locate the GM .plt for a YYYYMMDD_HHMMSS stamp, or None."""
    parts = timestamp_str.split('_')
    if len(parts) != 2:
        return None
    date_part, time_part = parts
    matches = glob.glob(os.path.join(plt_dir, "*e{}-{}*.plt".format(date_part, time_part)))
    return matches[0] if matches else None


def collect_timestamps(seed_dirs):
    """Sorted union of YYYYMMDD_HHMMSS stamps across the per-surface seed dirs."""
    stamps = set()
    for seed_dir in seed_dirs:
        for p in glob.glob(os.path.join(seed_dir, 'boundary_*.csv')):
            m = re.match(r'boundary_(\d{8}_\d{6})\.csv', os.path.basename(p))
            if m:
                stamps.add(m.group(1))
    return sorted(stamps)


def load_b_field(plt_path):
    """Read a GM .plt and build the 'B' vector Calculator. Returns (reader, b_calc).

    Both proxies must be Delete()d by the caller when done with the timestep.
    """
    reader = VisItTecplotBinaryReader(registrationName='GM_Reader', FileName=[plt_path])
    reader.MeshStatus = ['global_field']
    reader.PointArrayStatus = ['B_x_nT', 'B_y_nT', 'B_z_nT', 'x', 'y', 'z']
    reader.UpdatePipeline()

    b_calc = Calculator(registrationName='B_Vector', Input=reader)
    b_calc.ResultArrayName = 'B'
    b_calc.Function = 'B_x_nT*iHat+B_y_nT*jHat+B_z_nT*kHat'
    return reader, b_calc


def trace_streamlines(b_calc, csv_path, trace_cfg):
    """Trace field lines from one seed CSV over a prepared B-field.

    Returns (tracer, points_source, csv_reader, poly, seed_poly) where poly /
    seed_poly are fetched VTK polydata for the shading seam. Proxies are left
    alive for rendering/cleanup by the caller; returns None on failure.
    """
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

        poly = _sm.Fetch(tracer)
        seed_poly = _sm.Fetch(points_source)
        return tracer, points_source, csv_reader, poly, seed_poly
    except Exception as e:
        print("  FAILED ({})".format(e))
        for proxy in (tracer, points_source, csv_reader):
            if proxy is not None:
                Delete(proxy)
        return None
