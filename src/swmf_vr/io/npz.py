"""Streaming reader for the radiation-belt flux NPZ.

The `flux` array is multi-GB (1683 x 51 x 48 x 12 x 12), so we stream it one
timestep frame at a time straight out of the zip rather than np.load-ing the
whole thing -- the SLURM --mem-per-cpu=4g budget depends on this. Do NOT replace
iter_flux_frames with random access (`data['flux'][t]`); that forces a full
decompress per frame or a full load.

Verbatim port of the v6 parse_npy_header / iter_flux_frames / extract_integrated_flux.
"""

import ast
import zipfile
import numpy as np


def parse_npy_header(file_obj):
    """Parse a .npy stream header, returning (shape, dtype)."""
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
    """Yield (timestep_index, frame) by streaming flux.npy sequentially."""
    with zipfile.ZipFile(npz_path, 'r') as zf:
        with zf.open('flux.npy') as f:
            shape, dtype = parse_npy_header(f)
            frame_elems = int(np.prod(shape[1:]))
            frame_bytes = frame_elems * dtype.itemsize
            for idx in range(shape[0]):
                raw = f.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                yield idx, np.frombuffer(raw, dtype=dtype).reshape(shape[1:])


def extract_integrated_flux(frame, e_idxs, a_idx, lat_dim=51, mlt_dim=48):
    """Sum flux over the selected energy channels at one pitch-angle index.

    Handles both array orientations (lat-major vs energy-major) seen in the data.
    """
    if frame.shape[0] == lat_dim and frame.shape[1] == mlt_dim:
        return np.sum(frame[:, :, e_idxs, a_idx], axis=2)
    if frame.shape[2] == lat_dim and frame.shape[3] == mlt_dim:
        return np.sum(frame[e_idxs, a_idx, :, :], axis=0)
    raise ValueError('Unexpected frame shape {}'.format(frame.shape))


def load_grids(npz_path):
    """Load the static grids + time array needed to drive extraction.

    Returns a dict with E_lvls, alpha_lvls, ro (equatorial, [0]), mlto ([0]) and
    the time array (falling back to frame indices if absent), mirroring how the
    v6 main() reads them.
    """
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
    return {
        'E_lvls': e_grid,
        'alpha_lvls': a_grid,
        'time': time_arr,
        'ro': ro_vals,
        'mlto': mlto_vals,
    }


def select_energy_alpha(e_grid, a_grid, energy_min_kev, target_alpha_val):
    """(e_idxs, a_idx) for energy >= threshold and nearest pitch angle."""
    e_idxs = np.where(e_grid >= energy_min_kev)[0]
    a_idx = int(np.abs(a_grid - target_alpha_val).argmin())
    return e_idxs, a_idx
