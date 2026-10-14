#!/usr/bin/env python3
"""Assemble a CRA delivery folder from a finished pipeline run.

    python3 scripts/make_delivery.py --out-root run_v4 --tag 20261014 \
        --movies run_v4/fp_cutaway_1e8.mp4 --obj-ts 20100406_000000 20100406_120000

Writes OUT_ROOT/deliver_<tag>/ with
    slices_csv/            one CSV per timestamp (delivery layout) + grid.json, zipped too
    cutaway_obj_gei/<ts>/  the cutaway OBJ packages (+ one zip per timestamp)
    fieldlines_gei/        obj_fp_{inner,outer,danger}_gei + gsm_transforms_fp.csv (zip)
    movies/                the cutaway movie(s)
    README.md              what the files are, frames, coordinate conventions, settings
    MANIFEST.txt           file counts and sizes
The run facts come from the environment (.env via slurm/env.sh): SWMF_RUN_DIR,
SWMF_BASE_ISO, SWMF_HANDOFF_RE, and the SEED/TRACE/CUTAWAY/SLICE_ARGS the PBS
jobs used, when they are set.
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import sys
import zipfile


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return '{:.1f} {}'.format(n, unit)
        n /= 1024.0
    return '{:.1f} PB'.format(n)


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for f in sorted(files):
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, os.path.dirname(src_dir)))


def stamp_range(stamps):
    def fmt(ts):
        return '{}-{}-{} {}:{} UT'.format(ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13])
    if not stamps:
        return 'none', 'none', 0
    return fmt(stamps[0]), fmt(stamps[-1]), len(stamps)


def run_description(run_dir):
    try:
        with open(os.path.join(run_dir, 'PARAM.in')) as f:
            lines = f.read().split('\n')
        for i, l in enumerate(lines):
            if l.startswith('#DESCRIPTION'):
                return lines[i + 1].split('\t')[0].strip()
    except Exception:
        pass
    return '(no PARAM.in description found)'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', default=datetime.date.today().strftime('%Y%m%d'))
    p.add_argument('--csv-dir', default=None, help='default OUT_ROOT/fp_slices/csv_gt2_sparse')
    p.add_argument('--movies', nargs='*', default=[])
    p.add_argument('--obj-ts', nargs='*', default=[])
    p.add_argument('--no-zip', action='store_true')
    args = p.parse_args()

    out_root = os.path.abspath(args.out_root)
    csv_dir = os.path.abspath(args.csv_dir or os.path.join(out_root, 'fp_slices', 'csv_gt2_sparse'))
    dest = os.path.join(out_root, 'deliver_' + args.tag)
    os.makedirs(dest, exist_ok=True)
    run_dir = os.environ.get('SWMF_RUN_DIR', '(SWMF_RUN_DIR unset)')
    manifest = []

    # 1. slice CSVs
    csvs = sorted(glob.glob(os.path.join(csv_dir, 'slices_*.csv')))
    slice_stamps = [os.path.basename(c)[len('slices_'):-len('.csv')] for c in csvs]
    if csvs:
        d = os.path.join(dest, 'slices_csv')
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for c in csvs:
            shutil.copy2(c, d)
        gj = os.path.join(csv_dir, 'grid.json')
        if os.path.exists(gj):
            shutil.copy2(gj, d)
        if not args.no_zip:
            zip_dir(d, os.path.join(dest, 'slices_csv.zip'))
        manifest.append(('slices_csv/', len(csvs) + int(os.path.exists(gj)), dir_size(d)))
        print('slices_csv: {} files, {}'.format(len(csvs), human(dir_size(d))))
    else:
        print('WARNING: no slice CSVs in {}'.format(csv_dir))

    # 2. cutaway OBJ packages
    obj_done = []
    for ts in args.obj_ts:
        src = os.path.join(out_root, 'cutaway_obj_gei', ts)
        if not os.path.isdir(src):
            print('WARNING: no cutaway OBJ package for', ts)
            continue
        d = os.path.join(dest, 'cutaway_obj_gei', ts)
        if os.path.isdir(d):
            shutil.rmtree(d)
        shutil.copytree(src, d)
        if not args.no_zip:
            zip_dir(d, os.path.join(dest, 'cutaway_obj_gei', 'cutaway_obj_gei_{}.zip'.format(ts)))
        obj_done.append(ts)
        manifest.append(('cutaway_obj_gei/{}/'.format(ts), len(os.listdir(d)), dir_size(d)))
    print('cutaway OBJ packages:', ', '.join(obj_done) or 'none')

    # 3. GEI field lines
    fl = os.path.join(dest, 'fieldlines_gei')
    n_fl = 0
    for s in ('inner', 'outer', 'danger'):
        src = os.path.join(out_root, 'obj_fp_{}_gei'.format(s))
        if os.path.isdir(src):
            d = os.path.join(fl, 'obj_fp_{}_gei'.format(s))
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(src, d)
            n_fl += len(glob.glob(os.path.join(d, 'fieldlines_*.obj')))
    tr = os.path.join(out_root, 'gsm_transforms_fp.csv')
    if os.path.isdir(fl):
        if os.path.exists(tr):
            shutil.copy2(tr, fl)
        if not args.no_zip:
            zip_dir(fl, os.path.join(dest, 'fieldlines_gei.zip'))
        manifest.append(('fieldlines_gei/', n_fl, dir_size(fl)))
        print('fieldlines_gei: {} OBJ'.format(n_fl))

    # 4. movies
    mv = os.path.join(dest, 'movies')
    movies = [m for m in args.movies if os.path.exists(m)]
    if movies:
        os.makedirs(mv, exist_ok=True)
        for m in movies:
            shutil.copy2(m, mv)
        manifest.append(('movies/', len(movies), dir_size(mv)))
    print('movies:', ', '.join(os.path.basename(m) for m in movies) or 'none')

    # 5. README
    first, last, n = stamp_range(slice_stamps)
    fl_stamps = sorted(os.path.basename(x)[len('fieldlines_'):-len('.obj')]
                       for x in glob.glob(os.path.join(out_root, 'obj_fp_outer', 'fieldlines_*.obj')))
    ffirst, flast, fn = stamp_range(fl_stamps)
    env = {k: os.environ.get(k, '(unset)') for k in
           ('SWMF_BASE_ISO', 'SWMF_HANDOFF_RE', 'SEED_ARGS', 'TRACE_ARGS', 'CUTAWAY_ARGS', 'SLICE_ARGS', 'CSV_ARGS')}
    readme = """# SWMF-VR delivery {tag}

Run: `{run_dir}`
{desc}
Frame 0 (SWMF_BASE_ISO): {base}. Analytic dipole below {handoff} Re, BATSRUS trace above.
Made with SWMF-VR commit `{commit}` on {host}, {date}.

## slices_csv/  (for the 3D texture)

{n} timestamps, {first} to {last}, one CSV per timestamp (`slices_<ts>.csv`) with all 24
meridional half-planes stacked; `azimuth_deg` is the cut (0 = SM +x = noon, MLT 12; azimuth
increases toward dusk: 90 = MLT 18, 180 = midnight, 270 = dawn). Positions are in GEI; the
per-frame SM -> GEI rotation and the static bounding box are in `grid.json`. Flux columns are
the > 2 MeV omnidirectional electron flux (cm^-2 s^-1); rows where it is 0 are omitted
(`--drop-zeros`) and the header of each file records the dense grid and the rows kept.
Scripts: `scripts/fp_slices.py` ({slice_args}) then `scripts/fp_slices_csv.py` ({csv_args}).

## cutaway_obj_gei/<ts>/  (the half-donut, for the VR app)

Timestamps: {obj_ts}. Each folder: `cutaway_<ts>.obj` (groups outer / inner / cutface, texture
coordinate `vt` against `ramp.png`), `cutaway_<ts>_vcolor.obj` (per-vertex colour variant),
`material.mtl`, `colormap.json` (the flux -> colour table), `README.txt`. Coordinates GEI.
Script: `scripts/fp_cutaway_obj.py --frame gei` with the colour range in `colormap.json`.

## fieldlines_gei/  (the belt boundary shells as field lines)

{fn} frames, {ffirst} to {flast}: `obj_fp_outer_gei`, `obj_fp_inner_gei`, `obj_fp_danger_gei`
(outer edge, inner edge, flux peak), one OBJ per frame (`fieldlines_<ts>.obj`), 48 lines (one
per MLT sector) in GEI; `gsm_transforms_fp.csv` holds the GSM -> GEO/GEI matrices and the dipole
axis per frame. Belt edges: {seed_args}; tracing: {trace_args}.

## movies/

Cutaway movie(s): noon-midnight cut, camera on the dawn side, dusk half drawn, GSM scene
axes with the GEI triad of each frame, {cutaway_args}.

## Conventions

* Scene/cut: GSM (+x to the Sun, +z north); the cut plane y = 0 is the noon-midnight meridian,
  shared by GSM and SM. Delivered geometry is in GEI (rotate with the per-frame matrices).
* Flux: CIMI (or RBE) electron flux at the northern foot point of the BATSRUS field line
  through each point, Liouville-mapped to the local |B|; omnidirectional = 4 pi integral over
  the local pitch-angle distribution. Units cm^-2 s^-1 (> 2 MeV band).
* Only closed field lines carry flux; open lines and points beyond the model's last closed
  line are empty.
"""
    commit = '(unknown)'
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    readme = readme.format(
        tag=args.tag, run_dir=run_dir, desc=run_description(run_dir), base=env['SWMF_BASE_ISO'],
        handoff=env['SWMF_HANDOFF_RE'], commit=commit, host=os.uname().nodename,
        date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        n=n, first=first, last=last, slice_args=env['SLICE_ARGS'], csv_args=env['CSV_ARGS'],
        obj_ts=', '.join(obj_done) or 'none', fn=fn, ffirst=ffirst, flast=flast,
        seed_args=env['SEED_ARGS'], trace_args=env['TRACE_ARGS'], cutaway_args=env['CUTAWAY_ARGS'])
    with open(os.path.join(dest, 'README.md'), 'w') as f:
        f.write(readme)

    with open(os.path.join(dest, 'MANIFEST.txt'), 'w') as f:
        f.write('delivery {}  ({})\n'.format(args.tag, datetime.datetime.now().isoformat()))
        for name, count, size in manifest:
            f.write('{:40s} {:6d} files  {}\n'.format(name, count, human(size)))
        for z in sorted(glob.glob(os.path.join(dest, '*.zip')) + glob.glob(os.path.join(dest, '*', '*.zip'))):
            f.write('{:40s}         {}\n'.format(os.path.relpath(z, dest), human(os.path.getsize(z))))
    print('delivery folder: {} ({})'.format(dest, human(dir_size(dest))))


if __name__ == '__main__':
    main()
