"""Single source of truth for pipeline parameters.

Replaces the constants that were hardcoded at the top of every v1-v6 script. A
default-constructed CRAConfig reproduces the v6 campaign exactly, so the
consolidated package is behavior-preserving out of the box.

Implemented with typing.NamedTuple (not dataclasses): the system python here is
3.6.8, which predates the stdlib `dataclasses` module, and the pure layer must
import under both 3.6.8 and pvbatch's 3.10. NamedTuple gives the same immutable,
typed, defaulted config and supports `._replace(...)` for CLI overrides.
"""

import os
from typing import NamedTuple, Optional, Tuple

# --- Color scalars for shading (see swmf_vr.shading / paraview.extract) -------
# PER-LINE scalars are read from the seed CSV and broadcast along each field
# line via the StreamTracer SeedIds cell array; ALONG-LINE scalars are computed
# per streamline vertex from geometry / the interpolated B field.
SEED_SCALARS = {'eqflux': 'Flux', 'eqflux_peak': 'Flux_peak', 'lshell': 'Equator_R'}
POINT_SCALARS = ('bmag', 'radius', 'lat')

# scalar -> (log?, default vmin, default vmax). CLI may override vmin/vmax/log.
SCALAR_DEFAULTS = {
    'eqflux':      (True,  1e0,  1e4),
    'eqflux_peak': (True,  1e0,  1e4),
    'lshell':      (False, 1.0,  6.0),
    'bmag':        (True,  1e1,  1e4),
    'radius':      (False, 1.0,  6.0),
    'lat':         (False, -90.0, 90.0),
}

# matplotlib cmap name -> ParaView builtin preset (preview screenshots only).
CMAP_PRESETS = {
    'inferno': 'Inferno (matplotlib)', 'viridis': 'Viridis (matplotlib)',
    'plasma':  'Plasma (matplotlib)',  'magma':   'Magma (matplotlib)',
}

# 2D flux-plot boundary colors, keyed by threshold value or 'max' (from v6).
PLOT_COLORS = {
    1: '#00ffff', 10: '#00ff00', 100: '#ffff00',
    250: '#ff8800', 300: '#ff6600', 500: '#ff0000', 'max': '#ff0000',
}

# Fixed epoch for geopack ut (distinct from the NPZ BASE_ISO frame origin).
UNIX_EPOCH_ISO = '1970-01-01T00:00:00'

# Campaign default paths (the live CRA layout). Resolved relative to this file
# so they survive the package moving, but overridable via PathConfig.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))           # .../CRA/SWMF-VR/src/swmf_vr
_REPO_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))         # .../CRA/SWMF-VR
_CRA_ROOT = os.path.dirname(_REPO_ROOT)                         # .../CRA

# Shared assets were relocated out of the (now archived) v1/ into neutral
# top-level dirs: CRA/data/ and CRA/tools/.
_DEFAULT_NPZ = os.path.join(_CRA_ROOT, 'data', '20240511_170000_e_fls.npz')
_DEFAULT_PLT_DIR = '/nfs/turbo/coe-tuija/shared/run_mothersday_ne/GM/IO2'
_DEFAULT_PVBATCH = os.path.join(
    _CRA_ROOT, 'tools',
    'ParaView-5.12.1-osmesa-MPI-Linux-Python3.10-x86_64', 'bin', 'pvbatch')


class PathConfig(NamedTuple):
    npz_path: str = _DEFAULT_NPZ
    plt_dir: str = _DEFAULT_PLT_DIR
    pvbatch: str = _DEFAULT_PVBATCH
    out_root: str = '.'          # base for flux_slices/, out_seeds_*/, obj_*/, ...


class FluxConfig(NamedTuple):
    energy_min_kev: float = 2000.0
    target_alpha_val: float = 1.0
    lat_dim: int = 51
    mlt_dim: int = 48


class TimeConfig(NamedTuple):
    base_iso: str = '2024-05-10T13:00:00'   # NPZ time-minutes origin


class SurfaceSpec(NamedTuple):
    name: str
    mode: str                    # 'crossing' | 'max'
    thresholds: Tuple = ()       # e.g. (10,) inner, (250,) outer; () for max
    search: str = ''             # 'inward' | 'outward' (crossing only)
    linestyle: str = '-'         # 2D flux-plot line style
    color: str = '#ffffff'       # 2D flux-plot line color
    rgb: Tuple = (1.0, 1.0, 1.0)  # 3D screenshot fallback color
    seed_subdir: str = ''        # out_seeds_inner / out_seeds_outer / out_seeds_danger
    obj_subdir: str = ''         # obj_inner / obj_outer / obj_max


# Order and values mirror v6 exactly (inner=green dashed, outer=orange solid,
# danger=red dotted). danger uses 'max' mode (per-MLT peak-flux radius).
DEFAULT_SURFACES = (
    SurfaceSpec('inner',  'crossing', (10,),  'outward', '--', '#00ff00',
                (0.0, 1.0, 0.0), 'out_seeds_inner',  'obj_inner'),
    SurfaceSpec('outer',  'crossing', (250,), 'inward',  '-',  '#ff8800',
                (1.0, 0.53, 0.0), 'out_seeds_outer',  'obj_outer'),
    SurfaceSpec('danger', 'max',      (),     '',        ':',  '#ff0000',
                (1.0, 0.0, 0.0), 'out_seeds_danger', 'obj_max'),
)


class ShadingConfig(NamedTuple):
    scalar: str = 'eqflux_peak'
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


def default_config() -> CRAConfig:
    """A config reproducing the v6 campaign exactly."""
    return CRAConfig()


def resolve_color_range(shading: ShadingConfig):
    """(log, vmin, vmax) for the active scalar, applying ShadingConfig overrides."""
    log_def, vmin_def, vmax_def = SCALAR_DEFAULTS[shading.scalar]
    log = log_def if shading.log is None else shading.log
    vmin = vmin_def if shading.vmin is None else shading.vmin
    vmax = vmax_def if shading.vmax is None else shading.vmax
    return log, vmin, vmax
