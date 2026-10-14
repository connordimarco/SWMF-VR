#!/usr/bin/env python3
"""Set up an SWMF (BATSRUS + RIM + CIMI) run directory for a new storm from a
finished template run, on Athena.

    python3 scripts/new_storm_run.py --template /nobackupp28/cdimarco/run_galaxy2010 \
        --dest /nobackupp28/cdimarco/run_<storm> \
        --start 2010-04-05T04:00 --end 2010-04-06T18:00 \
        --midl-csv /path/201004_32Re.csv [more monthly files if the span crosses a month] \
        --description "Storm X for CRA" [--nodes 3] [--walltime 36:00:00] [--sw-pad-hours 4]

What it does (nothing is submitted):
  * copies the template's inputs: PARAM.in (+ .init), job.athfe, the IE/IM data
    files, GM/IE/IM scripts, PostProc/Restart/Resubmit, midl_to_swmf.py; re-creates
    the symlinks (EIE, Param, SWMFDIR, SWMF.exe, GM/Param, GM/plots, IE/plots) and
    the empty output/restart directories; no outputs or restarts are copied;
  * rewrites #DESCRIPTION, #STARTTIME, #ENDTIME and #CPUTIMEMAX (walltime - 30 min)
    in PARAM.in; the steady-state sessions, grid and physics stay as the template;
  * converts the MIDL 32 Re monthly CSV(s) to IMF.dat for [start - pad, end + pad]
    with the template's midl_to_swmf.py and links IMF.dat to it;
  * rewrites the PBS header of job.athfe (-N, nodes/ranks, walltime) and writes a
    README.md recording what was done.
Then: `cd DEST; qsub job.athfe`, and when it finishes, point SWMF-VR's .env at
DEST (SWMF_RUN_DIR, SWMF_BASE_ISO = start, RBE_NPZ) and run the pbs/ jobs.
"""
import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys

COPY_FILES = ['PARAM.in', 'PARAM.in.init', 'job.athfe', 'PostProc.pl', 'Restart.pl',
              'Resubmit.pl', 'midl_to_swmf.py', 'magin_GEM.dat']
COPY_SUBDIR_FILES = {'IE': ('*.dat',), 'IM': ('*.dat', '*.fin'), 'GM': ('pIDL', 'pTEC'),
                     'IE/pION': None}
LINKS = ['EIE', 'Param', 'SWMFDIR', 'SWMF.exe', 'GM/Param', 'GM/plots', 'IE/plots']
EMPTY_DIRS = ['STDOUT', 'GM/IO2', 'GM/restartIN', 'GM/restartOUT', 'IM/plots', 'IM/restartIN',
              'IM/restartOUT', 'IE/ionosphere', 'IE/restartIN', 'IE/restartOUT']


def time_block(tag, dt):
    return ('#{}\n{}\t\t\tiYear\n{:02d}\t\t\tiMonth\n{:02d}\t\t\tiDay\n{:02d}\t\t\tiHour\n'
            '{:02d}\t\t\tiMinute\n{:02d}\t\t\tiSecond\n0.0\t\t\tFracSecond\n').format(
                tag, dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def replace_block(text, tag, new_block, n_lines):
    """Replace '#TAG' and the n_lines parameter lines after it."""
    pat = re.compile(r'^#' + tag + r'\n(?:[^\n]*\n){' + str(n_lines) + r'}', re.M)
    m = pat.search(text)
    if not m:
        sys.exit('PARAM.in: #{} block not found'.format(tag))
    return text[:m.start()] + new_block + text[m.end():]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--template', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--start', required=True, help='ISO, e.g. 2010-04-05T04:00')
    p.add_argument('--end', required=True)
    p.add_argument('--midl-csv', nargs='+', required=True, help='MIDL 32 Re monthly CSV(s)')
    p.add_argument('--description', required=True)
    p.add_argument('--nodes', type=int, default=3)
    p.add_argument('--ppn', type=int, default=256)
    p.add_argument('--walltime', default='36:00:00')
    p.add_argument('--sw-pad-hours', type=float, default=4.0,
                   help='solar wind file margin before start and after end')
    p.add_argument('--group', default='s3007')
    p.add_argument('--force', action='store_true', help='reuse an existing DEST')
    args = p.parse_args()

    t0 = datetime.datetime.strptime(args.start, '%Y-%m-%dT%H:%M')
    t1 = datetime.datetime.strptime(args.end, '%Y-%m-%dT%H:%M')
    if t1 <= t0:
        sys.exit('end must be after start')
    tpl, dest = os.path.abspath(args.template), os.path.abspath(args.dest)
    if os.path.exists(dest) and not args.force:
        sys.exit('{} exists (use --force to reuse)'.format(dest))
    os.makedirs(dest, exist_ok=True)

    # files
    for f in COPY_FILES:
        src = os.path.join(tpl, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, f))
    import glob
    for sub, patterns in COPY_SUBDIR_FILES.items():
        s_dir = os.path.join(tpl, sub)
        if not os.path.isdir(s_dir):
            continue
        d_dir = os.path.join(dest, sub)
        os.makedirs(d_dir, exist_ok=True)
        if patterns is None:
            for f in os.listdir(s_dir):
                full = os.path.join(s_dir, f)
                if os.path.isfile(full):
                    shutil.copy2(full, d_dir)
        else:
            for pat in patterns:
                for full in glob.glob(os.path.join(s_dir, pat)):
                    if os.path.isfile(full):
                        shutil.copy2(full, d_dir)
    for d in EMPTY_DIRS:
        os.makedirs(os.path.join(dest, d), exist_ok=True)
    for l in LINKS:
        src = os.path.join(tpl, l)
        if os.path.islink(src):
            target = os.readlink(src)
            link = os.path.join(dest, l)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(target, link)

    # solar wind
    pad = datetime.timedelta(hours=args.sw_pad_hours)
    sw0, sw1 = t0 - pad, t1 + pad
    sw_name = 'MIDL_32Re_{}_to_{}_swmf.dat'.format(sw0.strftime('%Y-%m-%dT%H%M'), sw1.strftime('%Y-%m-%dT%H%M'))
    parts = []
    for k, csv in enumerate(args.midl_csv):
        part = os.path.join(dest, '.sw_part{}.dat'.format(k))
        subprocess.check_call([sys.executable, os.path.join(dest, 'midl_to_swmf.py'), csv,
                               sw0.strftime('%Y-%m-%dT%H:%M'), sw1.strftime('%Y-%m-%dT%H:%M'), part])
        parts.append(part)
    with open(os.path.join(dest, sw_name), 'w') as out:
        header_done = False
        n_rows = 0
        for part in parts:
            with open(part) as f:
                lines = f.read().split('\n')
            try:
                i_start = lines.index('#START')
            except ValueError:
                sys.exit('{}: no #START line'.format(part))
            if not header_done:
                out.write('\n'.join(lines[:i_start + 1]) + '\n')
                header_done = True
            rows = [l for l in lines[i_start + 1:] if l.strip()]
            out.write('\n'.join(rows) + '\n')
            n_rows += len(rows)
            os.remove(part)
    imf = os.path.join(dest, 'IMF.dat')
    if os.path.lexists(imf):
        os.remove(imf)
    os.symlink(sw_name, imf)
    if n_rows < (sw1 - sw0).total_seconds() / 60 * 0.9:
        print('WARNING: solar wind file has {} rows for a {:.0f} min window'.format(
            n_rows, (sw1 - sw0).total_seconds() / 60))

    # PARAM.in
    pin = os.path.join(dest, 'PARAM.in')
    text = open(pin).read()
    text = replace_block(text, 'DESCRIPTION', '#DESCRIPTION\n{}\t\tStringDescription\n'.format(args.description), 1)
    text = replace_block(text, 'STARTTIME', time_block('STARTTIME', t0), 7)
    text = replace_block(text, 'ENDTIME', time_block('ENDTIME', t1), 7)
    h, m, s = (int(v) for v in args.walltime.split(':'))
    cpu_max = h + m / 60.0 + s / 3600.0 - 0.5
    text = replace_block(text, 'CPUTIMEMAX', '#CPUTIMEMAX\n{:.1f} hour\t\tCpuTimeMax\t! walltime {} - 30 min\n'.format(cpu_max, args.walltime), 1)
    open(pin, 'w').write(text)

    # job.athfe
    job = os.path.join(dest, 'job.athfe')
    jt = open(job).read()
    name = os.path.basename(dest).replace('run_', '')[:15]
    jt = re.sub(r'^#PBS -N .*$', '#PBS -N {}'.format(name), jt, flags=re.M)
    jt = re.sub(r'^#PBS -W group_list=.*$', '#PBS -W group_list={}'.format(args.group), jt, flags=re.M)
    jt = re.sub(r'^#PBS -l select=\d+:ncpus=\d+:mpiprocs=\d+:model=(\w+)$',
                '#PBS -l select={}:ncpus={}:mpiprocs={}:model=\\1'.format(args.nodes, args.ppn, args.ppn), jt, flags=re.M)
    jt = re.sub(r'^#PBS -l walltime=.*$', '#PBS -l walltime={}'.format(args.walltime), jt, flags=re.M)
    ranks = args.nodes * args.ppn
    jt = re.sub(r'mpiexec -n \d+ --ppn \d+', 'mpiexec -n {} --ppn {}'.format(ranks, args.ppn), jt)
    jt = re.sub(r'RANKS: \d+', 'RANKS: {}'.format(ranks), jt)
    open(job, 'w').write(jt)

    with open(os.path.join(dest, 'README.md'), 'w') as f:
        f.write('# {}\n\n{}\n\nSet up {} with scripts/new_storm_run.py from the template `{}`.\n'
                'Span {} to {} UT; solar wind `{}` ({} rows, MIDL 32 Re: {}); {} nodes x {} ranks, '
                'walltime {}, CPUTIMEMAX {:.1f} h. Grid, numerics and CIMI settings are the template\'s.\n\n'
                '    qsub job.athfe\n'.format(
                    os.path.basename(dest), args.description, datetime.date.today(), tpl,
                    args.start, args.end, sw_name, n_rows, ', '.join(args.midl_csv),
                    args.nodes, ranks, args.walltime, cpu_max))
    print('run directory ready: {}'.format(dest))
    print('  PARAM.in: {} -> {}; CPUTIMEMAX {:.1f} h; IMF.dat -> {} ({} rows)'.format(
        args.start, args.end, cpu_max, sw_name, n_rows))
    print('  job.athfe: {} nodes x {} = {} ranks, walltime {}'.format(args.nodes, args.ppn, ranks, args.walltime))
    print('  next: cd {} && qsub job.athfe   (TestParam.pl first if PARAM.in was hand-edited)'.format(dest))


if __name__ == '__main__':
    main()
