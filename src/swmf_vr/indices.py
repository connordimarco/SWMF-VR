"""Geomagnetic-index time series from the SWMF run logs, for the context strips
under the flux maps. Pure python (system python3 or pvbatch).

The geoindex and GM logs share the SWMF log format (title line, column-name line,
then data); the IMF input .dat is its own columnar format. The run was restarted
at 17:00, so each log comes in a 13:00 segment and a 17:00 segment -- they're
stitched with the restart winning the overlap.

  Bz  <- mothersday2_IMF.dat   (solar-wind driver)
  AL  <- geoindex_*.log        (auroral electrojet / substorms)
  Dst <- log_*.log (dst_sm)    (ring current / storm)
"""

import datetime

import numpy as np


def _read_swmf_log(path):
    """(datetimes, {col: float array}) from one SWMF log. Line 0 is a title, line
    1 the column names, line 2+ the data."""
    with open(path) as f:
        lines = f.readlines()
    if len(lines) < 3:
        return [], {}
    names = lines[1].split()
    idx = {n: i for i, n in enumerate(names)}
    keys = ('year', 'mo', 'dy', 'hr', 'mn', 'sc')
    if not all(k in idx for k in keys):
        return [], {}
    dts, cols = [], {n: [] for n in names}
    for ln in lines[2:]:
        p = ln.split()
        if len(p) < len(names):
            continue
        try:
            dt = datetime.datetime(*[int(p[idx[k]]) for k in keys])
        except (ValueError, IndexError):
            continue
        dts.append(dt)
        for n in names:
            try:
                cols[n].append(float(p[idx[n]]))
            except ValueError:
                cols[n].append(np.nan)
    return dts, {n: np.array(v) for n, v in cols.items()}


def series_from_logs(paths, col):
    """Stitch SWMF-log segments (later path wins the overlap) -> sorted (dts, vals)."""
    merged = {}
    for path in paths:
        dts, cols = _read_swmf_log(path)
        if col in cols:
            for dt, v in zip(dts, cols[col]):
                merged[dt] = v
    items = sorted(merged.items())
    return [t for t, _ in items], np.array([v for _, v in items])


def imf_series(path, col_idx=9):
    """One column from the IMF .dat (y mo dy hr mn sc msc bx by bz vx vy vz ...);
    col 9 = Bz. Data starts after the #START line."""
    dts, vals = [], []
    started = False
    with open(path) as f:
        for ln in f:
            if ln.lstrip().startswith('#START'):
                started = True
                continue
            if not started or ln.lstrip().startswith('#'):
                continue
            p = ln.split()
            if len(p) <= col_idx:
                continue
            try:
                dt = datetime.datetime(int(p[0]), int(p[1]), int(p[2]),
                                       int(p[3]), int(p[4]), int(p[5]))
                v = float(p[col_idx])
            except (ValueError, IndexError):
                continue
            dts.append(dt)
            vals.append(v)
    return dts, np.array(vals)


def build(out_npz, base_iso, t_end_h, gm_logs, geo_logs, imf_path):
    """Write Bz/AL/Dst series (x in hours since base, clipped to the run) to an
    .npz the plot can drop in. Returns the dict that was saved."""
    base = datetime.datetime.strptime(base_iso, '%Y-%m-%dT%H:%M:%S')

    def hours(dts):
        return np.array([(d - base).total_seconds() / 3600.0 for d in dts])

    raw = {'bz': imf_series(imf_path),
           'al': series_from_logs(geo_logs, 'AL'),
           'dst': series_from_logs(gm_logs, 'dst_sm')}
    out = {'base_iso': base_iso, 't_end': float(t_end_h)}
    for key, (dts, v) in raw.items():
        t = hours(dts)
        m = (t >= -0.5) & (t <= t_end_h + 0.5) & np.isfinite(v)
        out[key + '_t'], out[key + '_v'] = t[m], v[m]
    np.savez(out_npz, **out)
    return out
