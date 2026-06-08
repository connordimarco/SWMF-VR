"""Reading and writing: the flux NPZ, seed CSVs, OBJ vertices, time stamps.

The flux array is multi-GB, so iter_flux_frames streams it one timestep at a
time -- don't swap it for data['flux'][t], that loads the whole thing.
"""

import os
import re
import ast
import zipfile
import datetime
from collections import defaultdict

import numpy as np

BASE_ISO = '2024-05-10T13:00:00'
EPOCH = datetime.datetime(1970, 1, 1)

SEED_HEADER = 'Seed_X_Re,Seed_Y_Re,Seed_Z_Re,Equator_R,Equator_MLT,Flux,Flux_peak'


# --- flux NPZ ---------------------------------------------------------------

def parse_npy_header(file_obj):
    magic = file_obj.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError('Invalid NPY stream')
    major = file_obj.read(1)[0]
    _ = file_obj.read(1)[0]
    if major == 1:
        header_len = int.from_bytes(file_obj.read(2), 'little')
    elif major in (2, 3):
        header_len = int.from_bytes(file_obj.read(4), 'little')
    else:
        raise ValueError('Unsupported NPY version {}'.format(major))
    header = file_obj.read(header_len).decode('latin1')
    d = ast.literal_eval(header.strip())
    return d['shape'], np.dtype(d['descr'])


def iter_flux_frames(npz_path):
    """Yield (timestep_index, frame) streaming flux.npy in order."""
    with zipfile.ZipFile(npz_path, 'r') as zf:
        with zf.open('flux.npy') as f:
            shape, dtype = parse_npy_header(f)
            frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize
            for idx in range(shape[0]):
                raw = f.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                yield idx, np.frombuffer(raw, dtype=dtype).reshape(shape[1:])


def extract_integrated_flux(frame, e_idxs, a_idx, lat_dim=51, mlt_dim=48):
    """Sum flux over the chosen energy channels at one pitch angle. Handles both
    array orientations seen in the data."""
    if frame.shape[0] == lat_dim and frame.shape[1] == mlt_dim:
        return np.sum(frame[:, :, e_idxs, a_idx], axis=2)
    if frame.shape[2] == lat_dim and frame.shape[3] == mlt_dim:
        return np.sum(frame[e_idxs, a_idx, :, :], axis=0)
    raise ValueError('Unexpected frame shape {}'.format(frame.shape))


def load_grids(npz_path):
    """E_lvls, alpha_lvls, time, and the equatorial ro/mlto grids."""
    data = np.load(npz_path, allow_pickle=True)
    try:
        e_grid = data['E_lvls']
        a_grid = data['alpha_lvls']
        try:
            time_arr = data['time']
        except Exception:
            time_arr = np.arange(data['ro'].shape[0], dtype=float)
        ro_vals = data['ro'][0]
        mlto_vals = data['mlto'][0]
    finally:
        data.close()
    return {'E_lvls': e_grid, 'alpha_lvls': a_grid, 'time': time_arr,
            'ro': ro_vals, 'mlto': mlto_vals}


def select_energy_alpha(e_grid, a_grid, energy_min_kev, target_alpha_val):
    e_idxs = np.where(e_grid >= energy_min_kev)[0]
    a_idx = int(np.abs(a_grid - target_alpha_val).argmin())
    return e_idxs, a_idx


# --- seed CSVs --------------------------------------------------------------

def is_header_only(path):
    """No boundary this frame -> only the header line was written."""
    with open(path, 'r') as f:
        return len(f.readlines()) <= 1


def read_seed_csv(path):
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        return [], []
    cols = lines[0].split(',')
    rows = [[float(v) for v in ln.split(',')] for ln in lines[1:]]
    return cols, rows


# --- OBJ vertices -----------------------------------------------------------

def transform_obj(in_path, out_path, transform_fn):
    """Copy an OBJ, applying transform_fn(x,y,z) to every 'v x y z'. Everything
    else (vt / mtllib / usemtl / l) passes through, so the color survives."""
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            if line.startswith('v '):
                p = line.split()
                xt, yt, zt = transform_fn(float(p[1]), float(p[2]), float(p[3]))
                fout.write('v {:.10f} {:.10f} {:.10f}\n'.format(xt, yt, zt))
            else:
                fout.write(line)


# --- time -------------------------------------------------------------------

def base_datetime(base_iso=BASE_ISO):
    return datetime.datetime.strptime(base_iso, '%Y-%m-%dT%H:%M:%S')


def frame_datetime(time_val, base_dt):
    if isinstance(time_val, datetime.datetime):
        return time_val
    return base_dt + datetime.timedelta(minutes=float(time_val))


def timestamp_string(time_val, base_dt):
    return frame_datetime(time_val, base_dt).strftime('%Y%m%d_%H%M%S')


def dt_to_unix(dt):
    return (dt - EPOCH).total_seconds()


def parse_obj_timestamp(filename):
    m = re.search(r'fieldlines_(\d{8}_\d{6})\.obj', os.path.basename(filename))
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')


def resample_to_halfhour(src, dst, pattern):
    """Symlink the frame nearest each half-hour center (HH:15 / HH:45) into dst.
    pattern.group(2) is the YYYYMMDD_HHMMSS stamp."""
    os.makedirs(dst, exist_ok=True)
    files = {}
    for fname in os.listdir(src):
        m = pattern.match(fname)
        if m:
            files[datetime.datetime.strptime(m.group(2), '%Y%m%d_%H%M%S')] = fname
    if not files:
        print('  nothing matching in {}'.format(src))
        return []

    windows = defaultdict(list)
    for dt in files:
        windows[(dt.date(), dt.hour, 0 if dt.minute < 30 else 1)].append(dt)

    selected = []
    for (date, hour, half), dts in sorted(windows.items()):
        center = datetime.datetime(date.year, date.month, date.day, hour,
                                   15 if half == 0 else 45, 0)
        selected.append(files[min(dts, key=lambda dt: abs((dt - center).total_seconds()))])

    for fname in selected:
        link = os.path.join(dst, fname)
        if os.path.exists(link) or os.path.islink(link):
            os.remove(link)
        os.symlink(os.path.abspath(os.path.join(src, fname)), link)
    return selected
