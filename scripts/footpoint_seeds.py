#!/usr/bin/env python3
"""Find the belt boundaries in FOOT-POINT latitude and write seed CSVs for
the foot-point-seeded tracer (system python3, no ParaView).

  python3 scripts/footpoint_seeds.py [rank size] [--start N] [--max-frames N]
                                     [--stride N] [--force]
                                     [--npz PATH] [--out-root DIR]
                                     [--energy-min KEV] [--handoff RE]

Why this replaces the equatorial crawl
--------------------------------------
The old method walked the flux profile in EQUATORIAL radius (ro) and placed
seed points at z = 0, at radius ro. But RBE's "equatorial" quantities
(ro, mlto, bo) live at the minimum-B point of each field line, which leaves
the z = 0 plane in a storm-time field -- so the old seeds were on slightly
wrong field lines, and the traces then continued through the Earth on
non-physical inside-rBody field values.

The RBE flux grid is NATIVE to northern-ionosphere foot points: the grid
axes are magnetic latitude (lat_lvls, invariant latitude referenced to
r = 1 Re) by MLT (mlt_lvls). So instead we:

  1. walk the flux profile in foot-point LATITUDE, per MLT sector,
     to find where flux crosses each boundary threshold;
  2. only trust latitude rows below irm (RBE's own per-MLT index of the
     last closed field line -- rows at and above it hold stale copies);
  3. write the boundary FOOT POINTS out as seeds. Each foot point defines
     one dipole field line (L = 1/cos^2(lat)); where that line crosses the
     handoff sphere (2.5 Re) becomes the start point for the numeric
     BATSRUS trace. Lines with L <= 2.5 never reach the trace region and
     are drawn fully analytically.

This is the method Gabor asked for (2026-09-01): "we should be tracing
field lines from the north ionosphere, not from the equator."

Surfaces: inner = 10 (walking up in latitude from the equator side), outer =
10 (walking up from the flux PEAK -- see find_outer_boundary_lat for why not
down from the last closed line), danger = per-MLT flux peak; integrated
flux >= 2 MeV at pitch-angle sine 1.0. The outer threshold was lowered from the old
pipeline's 250 (decided 2026-09-01): at 250 the outer boundary hugged
L ~ 2.6-2.9 and barely moved during the storm, while at 10 it tracks the
belt's real extent (apex ~3 Re quiet, ~4-6 Re at storm peak). Threshold 1
was tried and rejected -- it latches onto flux holes inside the belt and
produces open lines at storm time.
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import config, data, dipole, frames
from geopack import geopack

FP_SEED_HEADER = ('Seed_X_Re,Seed_Y_Re,Seed_Z_Re,Foot_Lat_deg,Foot_MLT_hr,'
                  'L_shell,Trace_MHD,Flux,Flux_peak')

# Which latitude rows to trust: python rows 0 .. irm-1. irm comes from RBE
# as a Fortran (1-based) index of the last closed field line per MLT sector;
# checked 2026-09-01: the last distinct ro value sits at python row irm-1 and
# every row after it is a frozen copy of it.


def valid_latitude_rows(flux_column, irm_count, ro_column):
    """Indices of usable latitude rows in one MLT sector: below the
    last-closed-line cutoff, with a finite flux value, and with a RESOLVED
    equatorial mapping -- RBE's ro must keep increasing with latitude. Where
    ro repeats the previous row exactly, RBE's own field-line tracing has
    saturated (seen as long runs of ro = 8.46 or 15.12 near the boundary)
    and the flux stored there is not a resolved belt line."""
    rows = []
    previous_ro = -1.0
    for j in range(min(int(irm_count), len(flux_column))):
        if not np.isfinite(flux_column[j]):
            continue
        if not np.isfinite(ro_column[j]) or ro_column[j] <= previous_ro:
            continue
        rows.append(j)
        previous_ro = ro_column[j]
    return rows


def interpolate_crossing_latitude(threshold, flux_below, flux_above,
                                  lat_below_deg, lat_above_deg):
    """Linear interpolation of the latitude where flux crosses the
    threshold, between two adjacent grid rows."""
    if flux_above == flux_below:
        return lat_below_deg
    fraction = (threshold - flux_below) / (flux_above - flux_below)
    return lat_below_deg + fraction * (lat_above_deg - lat_below_deg)


def find_inner_boundary_lat(flux_slice, lat_lvls_deg, irm_counts, ro_slice,
                            threshold):
    """Per-MLT foot latitude of the belt's inner edge: walking UP in
    latitude (which walks outward in L), the first crossing from below the
    threshold to at-or-above it."""
    num_lats, num_mlts = flux_slice.shape
    result_lat_deg = np.full(num_mlts, np.nan)
    for m in range(num_mlts):
        rows = valid_latitude_rows(flux_slice[:, m], irm_counts[m], ro_slice[:, m])
        for k in range(len(rows) - 1):
            j_low, j_high = rows[k], rows[k + 1]
            if flux_slice[j_low, m] < threshold <= flux_slice[j_high, m]:
                result_lat_deg[m] = interpolate_crossing_latitude(
                    threshold, flux_slice[j_low, m], flux_slice[j_high, m],
                    lat_lvls_deg[j_low], lat_lvls_deg[j_high])
                break
    return result_lat_deg


# When walking outward from the peak, a dip below threshold that recovers
# within this many rows is treated as structure INSIDE the belt (e.g. a
# storm-time two-belt profile), not as the outer edge.
OUTER_GAP_ROWS = 2


def find_outer_boundary_lat(flux_slice, lat_lvls_deg, irm_counts, ro_slice,
                            threshold):
    """Per-MLT foot latitude of the belt's outer edge: starting at the flux
    PEAK (the main belt) and walking UP in latitude (outward in L), the
    first drop below the threshold that does not recover within
    OUTER_GAP_ROWS rows.

    Why from the peak and not down from the last closed line: RBE often
    carries detached high-latitude populations near its outer boundary
    (rows where its own ro mapping has saturated at ~8-15 Re) with flux just
    above threshold. Walking down from irm latches onto those and throws
    one sector's boundary out to L ~ 8-15 -- the "giant bump" lines. The
    belt's outer edge is where the population connected to the peak ends.

    If flux never drops below threshold before the last resolved row, the
    boundary is that last row (the belt extends to the last closed line)."""
    num_lats, num_mlts = flux_slice.shape
    result_lat_deg = np.full(num_mlts, np.nan)
    for m in range(num_mlts):
        rows = valid_latitude_rows(flux_slice[:, m], irm_counts[m], ro_slice[:, m])
        if len(rows) < 2:
            continue
        fluxes = [flux_slice[j, m] for j in rows]
        k_peak = int(np.argmax(fluxes))
        if fluxes[k_peak] < threshold:
            continue                       # no belt above threshold in this sector
        found = False
        for k in range(k_peak, len(rows) - 1):
            if fluxes[k] >= threshold and fluxes[k + 1] < threshold:
                # Does the flux come back within the gap allowance?
                look_ahead = fluxes[k + 2: k + 2 + OUTER_GAP_ROWS]
                if any(f >= threshold for f in look_ahead):
                    continue               # a dip inside the belt, keep walking
                result_lat_deg[m] = interpolate_crossing_latitude(
                    threshold, fluxes[k], fluxes[k + 1],
                    lat_lvls_deg[rows[k]], lat_lvls_deg[rows[k + 1]])
                found = True
                break
        if not found:
            result_lat_deg[m] = lat_lvls_deg[rows[-1]]
    return result_lat_deg


def find_danger_boundary_lat(flux_slice, lat_lvls_deg, irm_counts, ro_slice):
    """Per-MLT foot latitude of the flux PEAK (the danger surface), plus the
    peak flux value itself."""
    num_lats, num_mlts = flux_slice.shape
    result_lat_deg = np.full(num_mlts, np.nan)
    result_peak = np.full(num_mlts, np.nan)
    for m in range(num_mlts):
        rows = valid_latitude_rows(flux_slice[:, m], irm_counts[m], ro_slice[:, m])
        if not rows:
            continue
        fluxes = [flux_slice[j, m] for j in rows]
        k_max = int(np.argmax(fluxes))
        result_lat_deg[m] = lat_lvls_deg[rows[k_max]]
        result_peak[m] = fluxes[k_max]
    return result_lat_deg, result_peak


def write_footpoint_seed_csv(boundary_lat_deg, mlt_lvls_hr, flux_label,
                             flux_peak, handoff_radius_re, out_path):
    """One CSV row per MLT sector that has a boundary. The Seed_X/Y/Z point
    is in GSM: for lines that reach the handoff sphere it is the northern
    handoff crossing (where the numeric trace will start); for low lines
    that stay fully inside it is the line's apex (informational only --
    the tracer draws those analytically). geopack.recalc must already be
    set for this frame.

    Always writes the header, so a no-boundary frame leaves a header-only
    file (the tracer skips those).
    """
    with open(out_path, 'w') as f:
        f.write(FP_SEED_HEADER + '\n')
        for m in range(len(boundary_lat_deg)):
            foot_lat_deg = boundary_lat_deg[m]
            if not np.isfinite(foot_lat_deg):
                continue
            mlt_hr = float(mlt_lvls_hr[m])
            l_shell = dipole.l_shell_from_foot_latitude(foot_lat_deg)

            if l_shell > handoff_radius_re:
                # This line reaches the BATSRUS region: the seed is where
                # the dipole line crosses the handoff sphere, in the north.
                lat_at_handoff_deg = dipole.latitude_where_line_crosses_radius(
                    l_shell, handoff_radius_re)
                x_sm, y_sm, z_sm = dipole.sm_point_on_line(
                    l_shell, mlt_hr, lat_at_handoff_deg)
                trace_mhd = 1
            else:
                # Low line, fully dipolar: record the apex for reference.
                x_sm, y_sm, z_sm = dipole.sm_point_on_line(l_shell, mlt_hr, 0.0)
                trace_mhd = 0
            x_gsm, y_gsm, z_gsm = geopack.smgsm(x_sm, y_sm, z_sm, 1)

            flux_val = (float(flux_label[m]) if np.ndim(flux_label) > 0
                        else float(flux_label))
            peak_val = (float(flux_peak[m]) if np.isfinite(flux_peak[m])
                        else flux_val)
            f.write('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:d},'
                    '{:.6e},{:.6e}\n'.format(
                        x_gsm, y_gsm, z_gsm, foot_lat_deg, mlt_hr,
                        l_shell, trace_mhd, flux_val, peak_val))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('shard', nargs='*', type=int)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--force', action='store_true')
    p.add_argument('--npz', default=None)
    p.add_argument('--out-root', default=None)
    p.add_argument('--energy-min', type=float, default=None)
    p.add_argument('--handoff', type=float, default=dipole.HANDOFF_RADIUS_RE)
    args = p.parse_args()

    rank, size = (args.shard + [0, 1])[:2]

    cfg = config.CRAConfig()
    if args.npz:
        cfg = cfg._replace(paths=cfg.paths._replace(npz_path=os.path.abspath(args.npz)))
    if args.out_root:
        cfg = cfg._replace(paths=cfg.paths._replace(out_root=os.path.abspath(args.out_root)))
    if args.energy_min is not None:
        cfg = cfg._replace(flux=cfg.flux._replace(energy_min_kev=args.energy_min))

    out = cfg.paths.out_root
    seed_dirs = {}
    for s in cfg.surfaces:
        d = os.path.join(out, 'out_seeds_fp_' + s.name)
        os.makedirs(d, exist_ok=True)
        seed_dirs[s.name] = d

    # The grids. lat_lvls / mlt_lvls are the npz's native foot-point axes;
    # irm is small enough to load whole (it is constant along the latitude
    # axis, so one value per (time, MLT) is all we keep).
    g = data.load_grids(cfg.paths.npz_path)
    npz = np.load(cfg.paths.npz_path, allow_pickle=True)
    lat_lvls_deg = np.asarray(npz['lat_lvls'], dtype=float)
    mlt_lvls_hr = np.asarray(npz['mlt_lvls'], dtype=float)
    irm_per_time_mlt = np.asarray(npz['irm'][:, 0, :], dtype=float)
    cfg = cfg._replace(flux=cfg.flux._replace(lat_dim=len(lat_lvls_deg),
                                                mlt_dim=len(mlt_lvls_hr)))
    ro_all = np.asarray(npz['ro'], dtype=float)      # (T, L, M), ~33 MB
    npz.close()

    time_arr = g['time']
    e_idxs, a_idx = data.select_energy_alpha(
        g['E_lvls'], g['alpha_lvls'], cfg.flux.energy_min_kev, cfg.flux.target_alpha_val)
    base_dt = data.base_datetime(cfg.time.base_iso)
    print('energy channels >= {:.0f} keV: {}'.format(
        cfg.flux.energy_min_kev, g['E_lvls'][e_idxs].tolist()))
    print('handoff radius: {:.3f} Re; worker {}/{}'.format(args.handoff, rank, size))

    done = 0
    for t_idx, frame in data.iter_flux_frames(cfg.paths.npz_path):
        if t_idx >= len(time_arr):
            break
        if t_idx < args.start or t_idx % size != rank:
            continue
        if args.stride > 1 and ((t_idx - args.start) // size) % args.stride != 0:
            continue

        frame_dt = data.frame_datetime(time_arr[t_idx], base_dt)
        ts = frame_dt.strftime('%Y%m%d_%H%M%S')

        have = all(os.path.exists(os.path.join(seed_dirs[s.name],
                                               'boundary_{}.csv'.format(ts)))
                   for s in cfg.surfaces)
        if have and not args.force:
            done += 1
            continue

        flux_slice = data.extract_integrated_flux(
            frame, e_idxs, a_idx, cfg.flux.lat_dim, cfg.flux.mlt_dim)
        irm_counts = irm_per_time_mlt[t_idx, :]
        ro_slice = ro_all[t_idx]

        inner_lat = find_inner_boundary_lat(
            flux_slice, lat_lvls_deg, irm_counts, ro_slice, 10.0)
        outer_lat = find_outer_boundary_lat(
            flux_slice, lat_lvls_deg, irm_counts, ro_slice, 10.0)
        danger_lat, peak_flux = find_danger_boundary_lat(
            flux_slice, lat_lvls_deg, irm_counts, ro_slice)

        # The SM -> GSM rotation (dipole tilt) is time dependent; set it once
        # for this frame before converting any seed points.
        frames.recalc_for(frame_dt)

        write_footpoint_seed_csv(
            inner_lat, mlt_lvls_hr, 10.0, peak_flux, args.handoff,
            os.path.join(seed_dirs['inner'], 'boundary_{}.csv'.format(ts)))
        write_footpoint_seed_csv(
            outer_lat, mlt_lvls_hr, 10.0, peak_flux, args.handoff,
            os.path.join(seed_dirs['outer'], 'boundary_{}.csv'.format(ts)))
        write_footpoint_seed_csv(
            danger_lat, mlt_lvls_hr, peak_flux, peak_flux, args.handoff,
            os.path.join(seed_dirs['danger'], 'boundary_{}.csv'.format(ts)))

        done += 1
        if done % 25 == 0:
            print('  worker {}: {} frames done (latest {})'.format(rank, done, ts))
        if args.max_frames is not None and done >= args.max_frames:
            break

    print('worker {}: {} frames'.format(rank, done))


if __name__ == '__main__':
    main()
