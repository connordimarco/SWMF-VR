#!/usr/bin/env python3
"""RBE flux -> 3D voxel grid via BATSRUS foot points (pvbatch) -- NO tracing.

  "$PV_BATCH" scripts/footpoint_volume.py --ts 20240510_210000
       [--step RE] [--extent RE] [--quantity omni|perp] [--out-root DIR] [--force]

How this differs from volume3d.py (the old method)
--------------------------------------------------
The old volume stream-traced the field line of every lattice point (~33k
ParaView traces per frame, ~6 min) and looked RBE's flux up at the line's
z = 0 equatorial crossing -- an approximation, since RBE's flux actually
lives at the minimum-B point of the line (Austin, 2026-08-18 email chain).

This script uses the recipe Austin suggested instead: BATSRUS's own ray
tracer has ALREADY assigned every cell its ionospheric foot point, stored in
the 3D output as `theta_1_deg` / `phi_1_deg` (northern foot point) plus a
`Status` flag for open/closed. The RBE flux grid is NATIVE to those same
foot points (latN x mltN). So for each lattice point we simply:

  1. read its foot point (theta_1, phi_1) and open/closed Status from the
     resampled BATSRUS arrays -- no tracing;
  2. interpolate RBE's pitch-angle distribution and B_min (`bo`) on the
     regular foot-point grid at that (lat, MLT);
  3. Liouville/first-invariant map from B_min to the local |B|, exactly as
     before (mapping.adiabatic_omni / _perp).

Frames verified 2026-09-01 (probe at 48 known foot points): theta_1/phi_1
are SM latitude/longitude with `_1` = north, and MLT = phi_1/15 + 12. The
RBE grid is the same SM frame, so no rotation is applied anywhere. One known
~1 deg bias remains: BATSRUS labels foot points at its ionosphere radius
(~1.0157 Re) while the npz latitudes are invariant latitude at r = 1 -- a
fraction of one grid row, accepted for now ("start simple", Gabor).

Output format and file naming are IDENTICAL to volume3d.py so volume_gei.py
and the VR ingestion work unchanged -- only the parent directory differs
(volume_renders_fp/ instead of volume_renders/), so the two products can be
compared side by side and the old delivery is untouched.
"""

import os
import sys
import time
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paraview.simple import (VisItTecplotBinaryReader, TrivialProducer,
                             ResampleWithDataset, Delete)
from paraview import servermanager as _sm
from paraview import vtk as pvtk

from swmf_vr import config, render, mapping
import swmf_vr.data as data
from slice3d import BANDS
from volume3d import volume_seeds, write_csv

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# BATSRUS ray-tracer status: 3 = closed (both foot points on the body).
# Everything else (open one/both ends, body, unset) has no trapped population.
CLOSED_STATUS = 3.0


class RbeFootpointGrid:
    """Bilinear lookup of the RBE pitch-angle distribution and B_min on the
    native foot-point grid: regular in latitude (lat_lvls) x MLT (mlt_lvls,
    periodic). Latitude rows at and above irm (last closed line, per MLT
    sector) hold stale frozen values in the .fls output and are treated as
    off-grid."""

    def __init__(self, lat_lvls_deg, mlt_lvls_hr, flux_frame, bo_tesla,
                 irm_counts):
        self.lat_deg = np.asarray(lat_lvls_deg, dtype=float)      # (L,) ascending
        self.mlt_hr = np.asarray(mlt_lvls_hr, dtype=float)        # (M,) 0..23.5
        self.mlt_step_hr = float(self.mlt_hr[1] - self.mlt_hr[0])
        self.flux = flux_frame                                     # (L, M, E, A)
        self.bo_tesla = bo_tesla                                   # (L, M)
        # Highest USABLE latitude row per MLT sector: python rows 0..irm-1
        # are real (irm is Fortran 1-based; checked 2026-09-01).
        self.last_row = np.array([int(c) - 1 for c in irm_counts])

    def at(self, foot_lat_deg, foot_mlt_hr):
        """Interpolated (pad (E,A), bo) at a foot point, or (None, None) when
        the point is off the grid or in a sector's stale zone."""
        lat = self.lat_deg
        if not (lat[0] <= foot_lat_deg <= lat[-1]):
            return None, None
        j = int(np.searchsorted(lat, foot_lat_deg)) - 1
        j = max(0, min(j, len(lat) - 2))
        wj = (foot_lat_deg - lat[j]) / (lat[j + 1] - lat[j])

        m_float = (foot_mlt_hr % 24.0) / self.mlt_step_hr
        m = int(m_float) % len(self.mlt_hr)
        m_next = (m + 1) % len(self.mlt_hr)
        wm = m_float - int(m_float)

        # The four surrounding grid cells, dropping any that lie beyond that
        # sector's last closed line (their stored values are stale copies).
        corners = [(j,     m,      (1.0 - wj) * (1.0 - wm)),
                   (j + 1, m,      wj * (1.0 - wm)),
                   (j,     m_next, (1.0 - wj) * wm),
                   (j + 1, m_next, wj * wm)]
        pad_sum = np.zeros(self.flux.shape[2:], dtype=float)
        bo_sum = 0.0
        w_sum = 0.0
        for row, col, w in corners:
            if w <= 0.0 or row > self.last_row[col]:
                continue
            pad_sum += w * self.flux[row, col]
            bo_sum += w * self.bo_tesla[row, col]
            w_sum += w
        if w_sum < 0.5:          # mostly in the stale zone -> treat as off-grid
            return None, None
        return pad_sum / w_sum, bo_sum / w_sum


def sample_batsrus_at(plt_path, lattice_points):
    """Resample the BATSRUS arrays we need (B, Status, northern foot point)
    at the lattice points. Returns dict of numpy arrays, one value per point."""
    reader = producer = resample = None
    try:
        reader = VisItTecplotBinaryReader(FileName=[plt_path])
        reader.MeshStatus = ['global_field']
        reader.PointArrayStatus = ['B_x_nT', 'B_y_nT', 'B_z_nT', 'Status',
                                   'theta_1_deg', 'phi_1_deg']
        reader.UpdatePipeline()

        points = pvtk.vtkPoints()
        for x, y, z in lattice_points:
            points.InsertNextPoint(float(x), float(y), float(z))
        poly = pvtk.vtkPolyData()
        poly.SetPoints(points)
        producer = TrivialProducer()
        producer.GetClientSideObject().SetOutput(poly)
        producer.UpdatePipeline()

        resample = ResampleWithDataset(SourceDataArrays=reader,
                                       DestinationMesh=producer)
        resample.UpdatePipeline()
        out = _sm.Fetch(resample)
        pd = out.GetPointData()

        arrays = {}
        for i in range(pd.GetNumberOfArrays()):
            name = pd.GetArrayName(i)
            arr = pd.GetArray(i)
            arrays[name] = np.array([arr.GetValue(k)
                                     for k in range(arr.GetNumberOfTuples())])
        return arrays
    finally:
        for proxy in (resample, producer, reader):
            if proxy is not None:
                Delete(proxy)


def map_volume_footpoint(arrays, rbe_grid, band_ch, widths, sin_grid, quantity):
    """Flux at every lattice point from its foot point. Returns the (K, nbands)
    flux array (NaN = unmapped) and diagnostic counters."""
    valid_mask = arrays.get('vtkValidPointMask')
    status = arrays['Status']
    theta_1_deg = arrays['theta_1_deg']
    phi_1_deg = arrays['phi_1_deg']
    b_local_nt = np.sqrt(arrays['B_x_nT'] ** 2 + arrays['B_y_nT'] ** 2
                         + arrays['B_z_nT'] ** 2)

    K = len(status)
    flux = np.full((K, len(BANDS)), np.nan)
    n_closed = n_ongrid = n_mapped = 0
    b_ratio = []
    for k in range(K):
        if valid_mask is not None and valid_mask[k] == 0:
            continue                       # lattice point outside the mesh
        if status[k] != CLOSED_STATUS:
            continue                       # open line: no trapped particles
        n_closed += 1
        foot_lat_deg = theta_1_deg[k]
        foot_mlt_hr = (phi_1_deg[k] / 15.0 + 12.0) % 24.0
        pad, bo_tesla = rbe_grid.at(foot_lat_deg, foot_mlt_hr)
        if pad is None:
            continue                       # off the RBE grid / stale zone
        n_ongrid += 1
        b_s_tesla = b_local_nt[k] * 1e-9
        mapped_any = False
        for b, ch in enumerate(band_ch):
            if len(ch) == 0:
                continue
            j_eq = (pad[ch] * widths[ch, None]).sum(axis=0)   # (A,)
            if quantity == 'omni':
                val = mapping.adiabatic_omni(b_s_tesla, bo_tesla, j_eq, sin_grid)
            else:
                val = mapping.adiabatic_perp(b_s_tesla, bo_tesla, j_eq, sin_grid)
            if np.isfinite(val) and val > 0:
                flux[k, b] = val
                mapped_any = True
        if mapped_any:
            n_mapped += 1
        if bo_tesla and bo_tesla > 0:
            b_ratio.append(b_s_tesla / bo_tesla)   # local |B| / B_min, >= 1
    return flux, n_closed, n_ongrid, n_mapped, b_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True, help='timestamp YYYYMMDD_HHMMSS (one frame)')
    p.add_argument('--step', type=float, default=0.5, help='grid spacing (Re)')
    p.add_argument('--extent', type=float, default=10.0)
    p.add_argument('--quantity', choices=['omni', 'perp'], default='omni')
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--npz', default=None)
    p.add_argument('--plt-dir', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    cfg = config.CRAConfig()
    npz = os.path.abspath(args.npz) if args.npz else cfg.paths.npz_path
    plt_dir = os.path.abspath(args.plt_dir) if args.plt_dir else cfg.paths.plt_dir

    csv_dir = os.path.join(args.out_root, 'volume_renders_fp', 'csv')
    npz_dir = os.path.join(args.out_root, 'volume_renders_fp', 'npz')
    for d in (csv_dir, npz_dir):
        os.makedirs(d, exist_ok=True)
    ts = args.ts
    out_csv = os.path.join(csv_dir, 'volume_{}_bands.csv'.format(ts))
    out_npz = os.path.join(npz_dir, 'volume_{}_bands.npz'.format(ts))
    if not args.force and os.path.exists(out_csv) and os.path.exists(out_npz):
        print('[{}] exists -- skip'.format(ts)); return

    plt_path = render.find_matching_plt(plt_dir, ts)
    if not plt_path:
        print('[{}] no .plt -- skip'.format(ts)); return
    if os.path.getsize(plt_path) == 0:
        # 20240511_045900 is a truncated 0-byte file (ParaView dies on it).
        print('[{}] empty .plt -- skip'.format(ts)); return

    # Frame index == minutes after base (1-min cadence, same as volume3d).
    base_dt = data.base_datetime(cfg.time.base_iso)
    t = int(round((datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
                   - base_dt).total_seconds() / 60.0))
    zf = np.load(npz, allow_pickle=True)
    E_lvls = zf['E_lvls']
    sin_grid = np.asarray(zf['alpha_lvls'], dtype=float)
    n_frames = zf['bo'].shape[0]
    if not (0 <= t < n_frames):
        zf.close(); print('ts {} -> frame {} out of range'.format(ts, t)); return
    lat_lvls = np.asarray(zf['lat_lvls'], dtype=float)
    mlt_lvls = np.asarray(zf['mlt_lvls'], dtype=float)
    bo_tesla = np.asarray(zf['bo'][t], dtype=float)
    irm_counts = np.asarray(zf['irm'][t, 0, :], dtype=float)
    zf.close()

    flux_frame = mapping.read_flux_frame(npz, t)
    if flux_frame is None:
        print('[{}] flux frame {} missing'.format(ts, t)); return

    band_ch = [np.where((E_lvls >= lo) & (E_lvls < hi))[0] for _, lo, hi in BANDS]
    widths = mapping.channel_widths(E_lvls)
    seeds = volume_seeds(args.extent, args.step)
    rbe_grid = RbeFootpointGrid(lat_lvls, mlt_lvls, flux_frame, bo_tesla,
                                irm_counts)

    t0 = time.time()
    print('[{}] sampling {} lattice points (step={} Re, no tracing)'.format(
        ts, len(seeds), args.step))
    arrays = sample_batsrus_at(plt_path, seeds)
    flux, n_closed, n_ongrid, n_mapped, b_ratio = map_volume_footpoint(
        arrays, rbe_grid, band_ch, widths, sin_grid, args.quantity)

    write_csv(out_csv, seeds, flux, ts, args)
    np.savez(out_npz, points=seeds.astype(np.float32), flux=flux.astype(np.float32),
             labels=[b[0] for b in BANDS], ts=ts, quantity=args.quantity,
             step=args.step, extent=args.extent, frame='GSM')
    rat = ('  |B|/bo med {:.2f}'.format(float(np.median(b_ratio)))
           if b_ratio else '')
    print('[{}] {} closed, {} on RBE grid, {} mapped of {} points{}  '
          '({:.0f} s)  -> {}'.format(ts, n_closed, n_ongrid, n_mapped,
                                     len(seeds), rat, time.time() - t0, out_csv))


if __name__ == '__main__':
    main()
