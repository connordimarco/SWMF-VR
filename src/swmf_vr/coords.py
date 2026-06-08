"""CAP 4 -- rotating into GEI.

GSM -> GEO/GEI coordinate transforms via geopack, the per-timestep rotation
matrices exported for the VR app, and the magnetic dipole axis in GEI. Verbatim
port of v5/export_gsm_transforms.py + v5/transform_obj_coords.py.

geopack.recalc(ut) is GLOBAL MUTABLE STATE: it sets the dipole orientation for
the given epoch, and every smgsm/geogsm/geigeo call reads it. So:
  * call recalc_for(dt) (or geopack.recalc(ut)) ONCE per timestep, BEFORE any
    transform for that timestep;
  * never interleave timesteps within a process (process-level worker sharding
    keeps this safe).
"""

import numpy as np
from geopack import geopack

from .timeaxis import dt_to_unix


def recalc_for(dt):
    """Set geopack's global dipole state for datetime `dt`. Returns the ut used."""
    ut = dt_to_unix(dt)
    geopack.recalc(ut)
    return ut


def gsm_to_geo(x, y, z):
    """GSM -> GEO (ECEF). geogsm: j=1 is GEO->GSM, j=-1 is GSM->GEO."""
    return geopack.geogsm(x, y, z, -1)


def gsm_to_gei(x, y, z):
    """GSM -> GEI (~EME2000/J2000) via GSM -> GEO -> GEI.
    geigeo: j=1 is GEI->GEO, j=-1 is GEO->GEI."""
    xg, yg, zg = geopack.geogsm(x, y, z, -1)   # GSM -> GEO
    return geopack.geigeo(xg, yg, zg, -1)        # GEO -> GEI


def rotation_matrix(transform_fn):
    """3x3 R such that R @ v_gsm = v_target, built from the transformed unit
    vectors. Assumes recalc_for has been called for the current timestep."""
    cols = [
        np.array(transform_fn(1.0, 0.0, 0.0)),
        np.array(transform_fn(0.0, 1.0, 0.0)),
        np.array(transform_fn(0.0, 0.0, 1.0)),
    ]
    return np.column_stack(cols)


def dipole_gei():
    """Magnetic dipole axis in GEI: SM z-hat (0,0,1) -> GSM -> GEO -> GEI.
    Assumes recalc_for has been called for the current timestep."""
    dipole_gsm = geopack.smgsm(0.0, 0.0, 1.0, 1)
    dipole_geo = geopack.geogsm(dipole_gsm[0], dipole_gsm[1], dipole_gsm[2], -1)
    return geopack.geigeo(dipole_geo[0], dipole_geo[1], dipole_geo[2], -1)


def transform_fn_for(target):
    """'geo' or 'gei' -> the matching GSM-> transform function."""
    if target == 'geo':
        return gsm_to_geo
    if target == 'gei':
        return gsm_to_gei
    raise ValueError("target must be 'geo' or 'gei', got {!r}".format(target))
