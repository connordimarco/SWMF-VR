"""How things get colored: the per-vertex OBJ color baked into `vt` (+ the ramp
the VR app samples), and the 2D equatorial flux plot with boundaries drawn on.
"""

import os
import json
import math

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

from .config import PLOT_COLORS


# --- OBJ shading ------------------------------------------------------------

def scalar_to_u(value, log, vmin, vmax):
    """Map a scalar to U in [0,1] for the colormap ramp."""
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
    """Write polylines to OBJ with a per-vertex `vt` color coord. points is a
    list of (x,y,z), cells a list of 0-based index lists. Returns (umin, umax)."""
    umin, umax = 1.0, 0.0
    with open(out_path, 'w') as f:
        f.write('# field lines colored by {}\n'.format(scalar_name))
        f.write('mtllib material.mtl\nusemtl ramp\n')
        for (x, y, z) in points:
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(x, y, z))
        for v in point_vals:
            u = scalar_to_u(v, log, vmin, vmax)
            umin, umax = min(umin, u), max(umax, u)
            f.write('vt {:.5f} 0.5\n'.format(u))
        for cell in cells:
            f.write('l ' + ' '.join('{0}/{0}'.format(i + 1) for i in cell) + '\n')
    return umin, umax


def write_color_assets(obj_dir, cmap, scalar_name, log, vmin, vmax):
    """ramp.png (the colormap), material.mtl, and a colormap.json that says how
    U maps back to the real value."""
    import matplotlib.cm as cm
    ramp = os.path.join(obj_dir, 'ramp.png')
    rgb = cm.get_cmap(cmap)(np.linspace(0.0, 1.0, 256))[:, :3]
    img = np.repeat((rgb * 255).astype('uint8')[None, :, :], 8, axis=0)
    plt.imsave(ramp, img)

    with open(os.path.join(obj_dir, 'material.mtl'), 'w') as f:
        f.write('newmtl ramp\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd ramp.png\n')

    with open(os.path.join(obj_dir, 'colormap.json'), 'w') as f:
        json.dump({'scalar': scalar_name, 'cmap': cmap, 'vmin': vmin,
                   'vmax': vmax, 'log': log, 'ramp': 'ramp.png',
                   'mapping': ('U=0 -> vmin, U=1 -> vmax; '
                               + ('log10-scaled' if log else 'linear'))},
                  f, indent=2)


# --- 2D flux plot -----------------------------------------------------------

def plot_frame(ro_vals, mlto_vals, flux_slice, boundary_lines, surfaces, ts_str,
               out_path, radial_limits=(0, 4), title_suffix=''):
    """Polar log flux heatmap with the inner/outer/danger boundaries on top.
    boundary_lines is keyed (surface_name, label) -> per-MLT radii."""
    theta_rad = np.radians((mlto_vals - 12.0) * 15.0)

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    flux_plot = np.where(flux_slice > 0, flux_slice, 1e-10)
    marker_size = 54 if (radial_limits[1] - radial_limits[0]) <= 3.0 else 18
    pcm = ax.scatter(theta_rad.ravel(), ro_vals.ravel(), c=flux_plot.ravel(),
                     cmap='inferno', norm=LogNorm(vmin=1e-2, vmax=1e4),
                     marker='s', s=marker_size, linewidths=0)

    earth_theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(earth_theta, np.ones(100), color='steelblue', zorder=10)
    ax.plot(earth_theta, np.ones(100), color='white', linewidth=0.8, zorder=11)

    cb = fig.colorbar(pcm, ax=ax, pad=0.12, shrink=0.6, anchor=(0.0, 0.0))
    cb.set_label('Integrated Flux (E >= 2 MeV)', fontsize=11)

    legend_handles = []
    mlt_centers = mlto_vals[0, :]
    theta_bnd = np.radians((mlt_centers - 12.0) * 15.0)
    for s in surfaces:
        labels = ['max'] if s.mode == 'max' else list(s.thresholds)
        for label in labels:
            color = PLOT_COLORS[label]
            r_arr = boundary_lines[(s.name, label)]
            mask = np.isfinite(r_arr)
            if np.sum(mask) < 3:
                continue
            idx = np.where(mask)[0]
            t_vals = np.append(theta_bnd[idx], theta_bnd[idx[0]])
            r_vals = np.append(r_arr[idx], r_arr[idx[0]])
            line, = ax.plot(t_vals, r_vals, color=color, linewidth=2.0,
                            linestyle=s.linestyle, alpha=0.95)
            line.set_path_effects([path_effects.Stroke(linewidth=3.2, foreground='black'),
                                   path_effects.Normal()])
            legend_handles.append(Line2D([0], [0], color=color, linewidth=2.5,
                                         linestyle=s.linestyle,
                                         label='{} @ {}'.format(s.name, label)))

    ax.set_ylim(*radial_limits)
    yticks = [t for t in [1, 1.5, 2, 2.5, 3, 3.5, 4]
              if radial_limits[0] <= t <= radial_limits[1]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:g} Re'.format(t) for t in yticks], fontsize=9)
    mlt_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    ax.set_xticks(np.radians((np.array(mlt_hours) - 12.0) * 15.0))
    ax.set_xticklabels(['{:02d} MLT'.format(h) for h in mlt_hours], fontsize=10)

    title = 'Equatorial Flux Slice (>=2 MeV integrated) - {}'.format(ts_str)
    if title_suffix:
        title += ' ({})'.format(title_suffix)
    ax.set_title(title, pad=20, fontsize=14, weight='bold')
    fig.legend(handles=legend_handles, loc='lower center', fontsize=9,
               ncol=4, framealpha=0.9, bbox_to_anchor=(0.45, -0.02))

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
