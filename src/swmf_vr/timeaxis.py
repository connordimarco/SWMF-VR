"""Time-axis handling: NPZ frame index <-> wall clock, and half-hour resampling.

Two epochs coexist and must stay distinct:
  * BASE_ISO + data['time'] minutes  -> NPZ frame index to wall clock.
  * Unix epoch (1970-01-01)          -> wall clock to geopack `ut` (see coords).
"""

import os
import re
import datetime
from collections import defaultdict

BASE_ISO = '2024-05-10T13:00:00'
EPOCH = datetime.datetime(1970, 1, 1)


def base_datetime(base_iso=BASE_ISO):
    return datetime.datetime.strptime(base_iso, '%Y-%m-%dT%H:%M:%S')


def frame_datetime(time_val, base_dt):
    """NPZ time value (minutes since base_dt, or an actual datetime) -> datetime."""
    if isinstance(time_val, datetime.datetime):
        return time_val
    return base_dt + datetime.timedelta(minutes=float(time_val))


def timestamp_string(time_val, base_dt):
    """YYYYMMDD_HHMMSS filename stamp for an NPZ time value."""
    return frame_datetime(time_val, base_dt).strftime('%Y%m%d_%H%M%S')


def dt_to_unix(dt):
    """Seconds since the Unix epoch (geopack `ut`)."""
    return (dt - EPOCH).total_seconds()


def parse_obj_timestamp(filename):
    """datetime from fieldlines_YYYYMMDD_HHMMSS.obj, or None."""
    m = re.search(r'fieldlines_(\d{8}_\d{6})\.obj', os.path.basename(filename))
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')


def resample_to_halfhour(src, dst, pattern):
    """Downsample a directory to half-hour resolution via symlinks.

    Verbatim port of v4/downsample_obj_renders.py: bucket files into half-hour
    windows, pick the frame nearest the window center (HH:15:00 / HH:45:00), and
    symlink it into `dst`. `pattern` is a compiled regex whose group(2) is the
    YYYYMMDD_HHMMSS stamp (group(1) the full filename). Symlinks (not copies)
    keep the storage cost near zero. src paths are absolute so links survive a
    differing cwd.
    """
    os.makedirs(dst, exist_ok=True)

    files = {}
    for fname in os.listdir(src):
        m = pattern.match(fname)
        if m:
            dt = datetime.datetime.strptime(m.group(2), '%Y%m%d_%H%M%S')
            files[dt] = fname

    if not files:
        print("  No matching files found in {}".format(src))
        return []

    windows = defaultdict(list)
    for dt in files:
        half = 0 if dt.minute < 30 else 1
        windows[(dt.date(), dt.hour, half)].append(dt)

    selected = []
    for (date, hour, half), dts in sorted(windows.items()):
        center_minute = 15 if half == 0 else 45
        center = datetime.datetime(date.year, date.month, date.day,
                                   hour, center_minute, 0)
        best = min(dts, key=lambda dt: abs((dt - center).total_seconds()))
        selected.append(files[best])

    for fname in selected:
        src_path = os.path.abspath(os.path.join(src, fname))
        dst_path = os.path.join(dst, fname)
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            os.remove(dst_path)
        os.symlink(src_path, dst_path)

    return selected
