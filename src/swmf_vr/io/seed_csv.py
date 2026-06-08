"""Seed-CSV schema helpers.

The byte-exact writer lives in swmf_vr.seeding (it owns the geopack SM->GSM
conversion and the %.4f/%.6e formatting that parity depends on). This module
holds the schema constant plus read / header-only helpers used by the tracing
layer and the parity harness.

Contract preserved from v6: a frame with no boundary still produces a file with
ONLY the header line. Stage B detects that via is_header_only and skips it.
"""

SEED_HEADER = ('Seed_X_Re,Seed_Y_Re,Seed_Z_Re,Equator_R,'
               'Equator_MLT,Flux,Flux_peak')

SEED_COLUMNS = SEED_HEADER.split(',')


def is_header_only(path):
    """True if the CSV has no data rows (header line only) -- no boundary found."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return len(lines) <= 1


def read_seed_csv(path):
    """Return (columns, rows) where rows is a list of lists of floats."""
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        return [], []
    columns = lines[0].split(',')
    rows = [[float(v) for v in ln.split(',')] for ln in lines[1:]]
    return columns, rows
