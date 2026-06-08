"""CAP 6 -- shading (pure half).

Maps a per-vertex scalar to a 1D colormap coordinate `U` baked into OBJ `vt`
texture coordinates, and writes the shared color assets (ramp.png + material.mtl
+ colormap.json) the VR app samples. This is the v6 coloring logic with the VTK
coupling removed: it operates on plain Python `points` / `cells` / `point_vals`
extracted by swmf_vr.paraview.extract, so the whole shading/OBJ-writing layer is
importable and testable under system python3.

Numeric formats (parity-critical): v at %.6f, vt at %.5f.
"""

import os
import json
import math


def scalar_to_u(value, log, vmin, vmax):
    """Map a raw scalar to U in [0,1] for the 1D colormap ramp."""
    if log:
        lo, hi = math.log10(max(vmin, 1e-30)), math.log10(max(vmax, 1e-30))
        x = math.log10(value) if value > 0 else lo
    else:
        lo, hi, x = vmin, vmax, value
    if hi <= lo:
        return 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def write_colored_obj(points, cells, point_vals, out_path,
                      scalar_name, log, vmin, vmax):
    """Write field-line polylines to OBJ with a per-vertex `vt` color coord.

    points       list of (x, y, z) vertex tuples.
    cells        list of polylines, each a list of 0-based vertex indices.
    point_vals   raw scalar per vertex (len == len(points)).
    Returns (umin, umax) actually written, for logging.

    Layout: 'v x y z' / 'vt U 0.5' per vertex, 'l v/vt ...' per polyline, with
    'mtllib material.mtl' + 'usemtl ramp' pointing at the shared ramp.
    """
    umin, umax = 1.0, 0.0
    with open(out_path, 'w') as f:
        f.write('# swmf_vr field lines, colored by '
                '{} (vt.u = colormap coordinate)\n'.format(scalar_name))
        f.write('mtllib material.mtl\n')
        f.write('usemtl ramp\n')
        for (x, y, z) in points:
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(x, y, z))
        for v in point_vals:
            u = scalar_to_u(v, log, vmin, vmax)
            umin, umax = min(umin, u), max(umax, u)
            f.write('vt {:.5f} 0.5\n'.format(u))
        for cell in cells:
            parts = ' '.join('{0}/{0}'.format(i + 1) for i in cell)
            f.write('l {}\n'.format(parts))
    return umin, umax


def write_color_assets(obj_dir, cmap, scalar_name, log, vmin, vmax):
    """Write ramp.png (256x8 colormap), material.mtl, colormap.json into obj_dir.

    The JSON sidecar records exactly how U maps back to the physical scalar so
    the VR team can label it. matplotlib imported lazily (Agg) to keep module
    import light and paraview-free.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    ramp_png = os.path.join(obj_dir, 'ramp.png')
    grad = np.linspace(0.0, 1.0, 256)
    rgb = cm.get_cmap(cmap)(grad)[:, :3]
    img = np.repeat((rgb * 255).astype('uint8')[None, :, :], 8, axis=0)
    plt.imsave(ramp_png, img)

    with open(os.path.join(obj_dir, 'material.mtl'), 'w') as f:
        f.write('newmtl ramp\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd ramp.png\n')

    with open(os.path.join(obj_dir, 'colormap.json'), 'w') as f:
        json.dump({'scalar': scalar_name, 'cmap': cmap,
                   'vmin': vmin, 'vmax': vmax, 'log': log, 'ramp': 'ramp.png',
                   'mapping': ('U=0 -> vmin, U=1 -> vmax; '
                               + ('log10-scaled' if log else 'linear'))},
                  f, indent=2)
