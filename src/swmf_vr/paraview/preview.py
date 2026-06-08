"""Scalar-colored screenshot preview (QA). pvbatch ONLY.

Renders the field lines with the SAME scalar/colormap/range baked into the OBJ
`vt`, so the PNG matches what the VR app gets. Verbatim port of v6
show_scalar_colored. Best-effort: returns the producer proxy, or None so the
caller can fall back to a flat per-surface color.
"""

from paraview.simple import (
    TrivialProducer, Show, ColorBy, GetColorTransferFunction,
)

from ..config import CMAP_PRESETS, resolve_color_range


def show_scalar_colored(render_view, poly, point_vals, shading_cfg):
    """Attach point_vals as a 'ColorScalar' array and color by it. Returns the
    TrivialProducer proxy (caller must Delete it) or None on failure."""
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

        disp = Show(tp, render_view)
        disp.Representation = 'Surface'
        ColorBy(disp, ('POINTS', 'ColorScalar'))
        log, vmin, vmax = resolve_color_range(shading_cfg)
        ctf = GetColorTransferFunction('ColorScalar')
        preset = CMAP_PRESETS.get(shading_cfg.cmap)
        if preset:
            ctf.ApplyPreset(preset, True)
        ctf.RescaleTransferFunction(vmin, vmax)
        if log:
            ctf.MapControlPointsToLogSpace()
            ctf.UseLogScale = 1
        disp.SetScalarBarVisibility(render_view, False)
        return tp
    except Exception as e:
        print("    (preview coloring failed: {})".format(e))
        return None
