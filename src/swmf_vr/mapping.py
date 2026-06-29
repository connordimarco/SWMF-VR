"""Adiabatic mapping of CIMI equatorial flux onto a point of a closed field line.

CIMI solves the bounce-averaged distribution on the equator, so its native output
is a 2D (equatorial) map of directional flux j_eq(E, alpha_eq) per field line. To
put that flux at an arbitrary point s on the same (closed) line:

  * first invariant + energy conservation remap the pitch angle:
        sin(alpha_eq) = sin(alpha_local) * sqrt(B_eq / B_s)
  * Liouville keeps the directional flux invariant along the line, so the local
    distribution is just the equatorial one folded by B.

Omnidirectional flux integrates the local pitch-angle cone; perpendicular flux is
the single local-90deg lookup. Pure python (no ParaView) so it is testable and
runs under either interpreter.
"""

import numpy as np


def adiabatic_omni(B_s, B_eq, j_eq, sin_alpha_grid, n_local=90):
    """Omnidirectional flux at a point with field B_s, on a line whose equatorial
    field is B_eq, given the equatorial directional flux j_eq sampled at
    sin_alpha_grid (equatorial pitch angle, in sin units, ascending). One energy."""
    if not (B_eq > 0 and B_s > 0) or not np.all(np.isfinite(j_eq)):
        return np.nan
    B_s = max(B_s, B_eq)                          # equator is the min |B|; guard noise
    ratio = np.sqrt(B_eq / B_s)
    a = np.linspace(0.0, np.pi / 2.0, n_local)
    sin_eq = np.sin(a) * ratio                    # local alpha -> equatorial alpha
    j = np.interp(sin_eq, sin_alpha_grid, j_eq)   # flat beyond the ~81deg PAD cap
    return float(4.0 * np.pi * np.trapz(j * np.sin(a), a))   # 2pi azimuth x 2 hemispheres


def adiabatic_perp(B_s, B_eq, j_eq, sin_alpha_grid):
    """Local-90deg (perpendicular) directional flux at the point."""
    if not (B_eq > 0 and B_s > 0) or not np.all(np.isfinite(j_eq)):
        return np.nan
    B_s = max(B_s, B_eq)
    sin_eq = np.sqrt(B_eq / B_s)                  # alpha_local = 90deg
    return float(np.interp(sin_eq, sin_alpha_grid, j_eq))


def channel_widths(E_lvls):
    """Per-channel energy widths (keV) for integrating j over an energy band,
    using geometric-mean edges (the channels are ~log-spaced)."""
    E = np.asarray(E_lvls, float)
    edges = np.empty(len(E) + 1)
    edges[1:-1] = np.sqrt(E[:-1] * E[1:])
    edges[0] = E[0] ** 2 / edges[1]
    edges[-1] = E[-1] ** 2 / edges[-2]
    return np.diff(edges)


class CimiEquator:
    """Inverse-distance lookup of the CIMI equatorial PAD by (r_eq, MLT_eq).

    ro / mlto / flux_frame / bo are one time slice: ro,mlto,bo are (L, M) and
    flux_frame is (L, M, E, A). MLT convention: noon (+X) = 12, dusk (+Y) = 18.
    Interpolating over the k nearest cells (weight 1/d) instead of snapping to
    one removes the salt-and-pepper holes a hard nearest-cell cutoff leaves.
    """

    def __init__(self, ro, mlto, flux_frame, bo=None, dmax=1.0, k=8):
        from scipy.spatial import cKDTree
        ang = np.radians((mlto - 12.0) * 15.0)
        gx, gy = ro * np.cos(ang), ro * np.sin(ang)
        valid = (np.isfinite(ro) & np.isfinite(mlto) & (ro >= 1.0)
                 & np.isfinite(flux_frame).all(axis=(2, 3))
                 & (flux_frame.sum(axis=(2, 3)) > 0))
        self.tree = cKDTree(np.column_stack([gx[valid], gy[valid]]))
        self.pad = flux_frame[valid]                       # (Ncell, E, A)
        self.bo = bo[valid] if bo is not None else None    # (Ncell,)
        self.dmax = dmax
        self.k = int(min(k, self.pad.shape[0]))

    def at(self, r_eq, mlt_eq):
        """Interpolated equatorial PAD (E, A) and B_eq at (r_eq, MLT_eq), or
        (None, None) if the footpoint falls outside the CIMI grid."""
        ang = np.radians((mlt_eq - 12.0) * 15.0)
        d, j = self.tree.query([r_eq * np.cos(ang), r_eq * np.sin(ang)], k=self.k)
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        if d[0] > self.dmax:
            return None, None
        w = 1.0 / (d + 1e-6); w /= w.sum()
        pad = (self.pad[j] * w[:, None, None]).sum(axis=0)            # (E, A)
        bo = float((self.bo[j] * w).sum()) if self.bo is not None else None
        return pad, bo


def read_flux_frame(npz_path, t):
    """Stream the flux NPZ and return frame t (shape L,M,E,A), or None."""
    from .data import iter_flux_frames
    for idx, fr in iter_flux_frames(npz_path):
        if idx == t:
            return fr
    return None
