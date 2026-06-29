"""Knobs for a run -- defaults reproduce the campaign we've been working with.

NamedTuples (not dataclasses) because the system python here is 3.6. Override a
field with `._replace(...)`.
"""

import os
from typing import NamedTuple, Optional, Tuple

# Color scalars. Per-line ones come from the seed CSV and paint the whole line;
# along-line ones are computed per vertex.
SEED_SCALARS = {'eqflux': 'Flux', 'eqflux_peak': 'Flux_peak', 'lshell': 'Equator_R'}
POINT_SCALARS = ('bmag', 'radius', 'lat')

# Measured global max integrated (>=2 MeV) flux over the run, so the brightest
# color is the actual peak flux instead of an arbitrary ceiling.
FLUX_MAX = 15875.0

# scalar -> (log?, vmin, vmax). CLI can override vmin/vmax/log.
SCALAR_DEFAULTS = {
    'eqflux':      (True,  1e0,  FLUX_MAX),
    'eqflux_peak': (True,  1e0,  FLUX_MAX),
    'lshell':      (False, 1.0,  6.0),
    'bmag':        (True,  1e1,  1e4),
    'radius':      (False, 1.0,  6.0),
    'lat':         (False, -90.0, 90.0),
}

# matplotlib cmap -> ParaView preset, just for the preview screenshot.
CMAP_PRESETS = {
    'inferno': 'Inferno (matplotlib)', 'viridis': 'Viridis (matplotlib)',
    'plasma':  'Plasma (matplotlib)',  'magma':   'Magma (matplotlib)',
}

# Boundary-line colors for the 2D flux plot, keyed by threshold or 'max'.
PLOT_COLORS = {
    1: '#00ffff', 10: '#00ff00', 100: '#ffff00',
    250: '#ff8800', 300: '#ff6600', 500: '#ff0000', 'max': '#ff0000',
}

_PKG = os.path.dirname(os.path.abspath(__file__))
_CRA = os.path.dirname(os.path.dirname(os.path.dirname(_PKG)))   # .../CRA

# Paths come from the environment (set by .env via slurm/env.sh); the literals are
# fallbacks so the package still works if .env wasn't sourced.
_NPZ = os.environ.get('CIMI_NPZ', os.path.join(_CRA, 'data', '20240511_170000_e_fls.npz'))
_PLT = os.path.join(
    os.environ.get('SWMF_RUN_DIR', '/nfs/turbo/coe-tuija/shared/run_mothersday_ne'),
    'GM', 'IO2')
_PVBATCH = os.environ.get('PV_BATCH', os.path.join(
    _CRA, 'tools', 'ParaView-5.12.1-osmesa-MPI-Linux-Python3.10-x86_64', 'bin', 'pvbatch'))


class PathConfig(NamedTuple):
    npz_path: str = _NPZ
    plt_dir: str = _PLT
    pvbatch: str = _PVBATCH
    out_root: str = '.'


class FluxConfig(NamedTuple):
    energy_min_kev: float = 2000.0
    target_alpha_val: float = 1.0
    lat_dim: int = 51
    mlt_dim: int = 48


class TimeConfig(NamedTuple):
    base_iso: str = '2024-05-10T13:00:00'   # frame 0 wall-clock; data['time'] is minutes after this


class SurfaceSpec(NamedTuple):
    name: str
    mode: str                    # 'crossing' | 'max'
    thresholds: Tuple = ()
    search: str = ''             # 'inward' | 'outward'
    linestyle: str = '-'
    color: str = '#ffffff'
    rgb: Tuple = (1.0, 1.0, 1.0)
    seed_subdir: str = ''
    obj_subdir: str = ''


# inner = green dashed (flux 10), outer = orange solid (flux 250), danger = red
# dotted (the per-MLT peak).
DEFAULT_SURFACES = (
    SurfaceSpec('inner',  'crossing', (10,),  'outward', '--', '#00ff00',
                (0.0, 1.0, 0.0), 'out_seeds_inner',  'obj_inner'),
    SurfaceSpec('outer',  'crossing', (250,), 'inward',  '-',  '#ff8800',
                (1.0, 0.53, 0.0), 'out_seeds_outer',  'obj_outer'),
    SurfaceSpec('danger', 'max',      (),     '',        ':',  '#ff0000',
                (1.0, 0.0, 0.0), 'out_seeds_danger', 'obj_max'),
)


class ShadingConfig(NamedTuple):
    scalar: str = 'eqflux'        # color each line by its equatorial flux
    cmap: str = 'inferno'
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    log: Optional[bool] = None


class CameraConfig(NamedTuple):
    view_size: Tuple = (1920, 1080)
    position: Tuple = (0.0, -50.0, 0.0)
    focal_point: Tuple = (0.0, 0.0, 0.0)
    view_up: Tuple = (0.0, 0.0, 1.0)
    background: Tuple = (0.0, 0.0, 0.0)


class TraceConfig(NamedTuple):
    max_streamline_length: float = 100.0
    integration_direction: str = 'BOTH'
    vectors: Tuple = ('POINTS', 'B')


class CRAConfig(NamedTuple):
    paths: PathConfig = PathConfig()
    flux: FluxConfig = FluxConfig()
    time: TimeConfig = TimeConfig()
    surfaces: Tuple = DEFAULT_SURFACES
    shading: ShadingConfig = ShadingConfig()
    camera: CameraConfig = CameraConfig()
    trace: TraceConfig = TraceConfig()


def resolve_color_range(shading):
    """(log, vmin, vmax) for the active scalar, with any overrides applied."""
    log_def, vmin_def, vmax_def = SCALAR_DEFAULTS[shading.scalar]
    log = log_def if shading.log is None else shading.log
    vmin = vmin_def if shading.vmin is None else shading.vmin
    vmax = vmax_def if shading.vmax is None else shading.vmax
    return log, vmin, vmax
