"""GSM -> GEO/GEI rotations via geopack. GEI is the frame the VR meshes go out in.

geopack.recalc(ut) is global state: call recalc_for(dt) once per frame before
any transform, and never interleave frames in one process.
"""

import numpy as np
from geopack import geopack

from .data import dt_to_unix


def recalc_for(dt):
    """Set geopack's dipole state for this frame. Returns the ut used."""
    ut = dt_to_unix(dt)
    geopack.recalc(ut)
    return ut


def gsm_to_geo(x, y, z):
    return geopack.geogsm(x, y, z, -1)


def gsm_to_gei(x, y, z):
    xg, yg, zg = geopack.geogsm(x, y, z, -1)   # GSM -> GEO
    return geopack.geigeo(xg, yg, zg, -1)        # GEO -> GEI


def rotation_matrix(transform_fn):
    """3x3 R with R @ v_gsm = v_target, built from the transformed unit vectors.
    Assumes recalc_for ran for this frame."""
    return np.column_stack([np.array(transform_fn(1.0, 0.0, 0.0)),
                            np.array(transform_fn(0.0, 1.0, 0.0)),
                            np.array(transform_fn(0.0, 0.0, 1.0))])


def dipole_gei():
    """Magnetic dipole axis in GEI: SM z-hat -> GSM -> GEO -> GEI."""
    d = geopack.smgsm(0.0, 0.0, 1.0, 1)
    d = geopack.geogsm(d[0], d[1], d[2], -1)
    return geopack.geigeo(d[0], d[1], d[2], -1)


def transform_fn_for(target):
    if target == 'geo':
        return gsm_to_geo
    if target == 'gei':
        return gsm_to_gei
    raise ValueError("target must be 'geo' or 'gei', got {!r}".format(target))
