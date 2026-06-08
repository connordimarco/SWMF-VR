"""The VTK -> plain-Python seam for shading.

Pulls geometry (points, polyline cells) and the chosen per-vertex color scalar
out of a fetched StreamTracer polydata, returning only plain Python lists. After
this, swmf_vr.shading writes the OBJ with no VTK dependency.

This module does not `import paraview` -- it only calls methods on the VTK
objects handed in -- but it lives in the paraview subpackage because those
objects only exist under pvbatch. Verbatim port of v6 compute_point_values,
extended to also return points + cells.
"""

import math

from ..config import SEED_SCALARS


def extract_geometry_and_scalar(poly, seed_poly, scalar):
    """Return (points, cells, point_vals) for a traced streamline polydata.

    points       list of (x, y, z) per vertex (streamline geometry, GSM).
    cells        list of polylines, each a list of 0-based vertex indices.
    point_vals   raw color scalar per vertex.

    PER-LINE scalars (eqflux/eqflux_peak/lshell) come from the seed point data,
    broadcast to every vertex of a line via the SeedIds cell array. ALONG-LINE
    scalars (bmag/radius/lat) are computed per vertex from geometry / the
    interpolated 'B' point array.
    """
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
            raise ValueError("seed column '{}' missing (re-run Stage A to add "
                             "Flux_peak)".format(SEED_SCALARS[scalar]))
        seed_vals = [seed_arr.GetValue(i)
                     for i in range(seed_arr.GetNumberOfTuples())]
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
        raise ValueError("unknown color scalar '{}'".format(scalar))
    return points, cells, vals
