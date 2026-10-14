#!/usr/bin/env python3
"""Make preplot-made BATSRUS Tecplot binaries (.plt, TDV112) readable by
ParaView's VisIt Tecplot reader, the way Austin's `3d__paraview_1_*.plt`
files are.

Raw `pTEC p` output (BATSRUS `3d VAR tcp` -> preplot) carries the zone title
"3D   N=... T=...", a chain of zone AUXDATA records (BLOCKS, BODYNUMDENSITY,
..., the content of Austin's `.aux` sidecars) and the variables "X [R]",
"Y [R]", "Z [R]". vtkVisItTecplotBinaryReader reports every such file as
"invalid" (every ParaView 5.12 - 6.0 build tried); Austin's files, which read
fine, have zone name `global_field`, no aux records and plain `x y z`. This
script rewrites ONLY the header section that way and copies the data section
byte for byte; the aux records are saved next to the file as `<name>.aux`.

  python3 scripts/plt_visit_fix.py IN.plt OUT.plt          # one file
  python3 scripts/plt_visit_fix.py --in-place DIR/*.plt    # originals -> DIR/preplot_orig/
"""
import argparse
import os
import shutil
import struct
import sys

ZONE_MARKER, EOH_MARKER = 299.0, 357.0
DS_AUX_MARKER, VAR_AUX_MARKER = 799.0, 899.0
RENAME = {'X [R]': 'x', 'Y [R]': 'y', 'Z [R]': 'z'}


class Reader(object):
    def __init__(self, f):
        self.f = f

    def i32(self):
        return struct.unpack('<i', self.f.read(4))[0]

    def f32(self):
        return struct.unpack('<f', self.f.read(4))[0]

    def f64(self):
        return struct.unpack('<d', self.f.read(8))[0]

    def string(self):
        out = []
        while True:
            c = self.i32()
            if c == 0:
                return ''.join(out)
            out.append(chr(c))


def enc_string(s):
    return b''.join(struct.pack('<i', ord(c)) for c in s) + struct.pack('<i', 0)


def parse_header(path):
    f = open(path, 'rb')
    r = Reader(f)
    magic = f.read(8)
    if magic != b'#!TDV112':
        raise ValueError('{}: not a TDV112 file ({!r})'.format(path, magic))
    byte_order, file_type = r.i32(), r.i32()
    title = r.string()
    nvar = r.i32()
    names = [r.string() for _ in range(nvar)]
    zones, aux_all = [], []
    while True:
        marker = r.f32()
        if abs(marker - ZONE_MARKER) < 1e-3:
            z = {'name': r.string(), 'parent': r.i32(), 'strand': r.i32(),
                 'soltime': r.f64(), 'unused': r.i32(), 'ztype': r.i32()}
            z['varloc'] = r.i32()
            z['varloc_list'] = [r.i32() for _ in range(nvar)] if z['varloc'] == 1 else []
            z['rawface'] = r.i32()
            z['misc'] = r.i32()
            if z['misc'] != 0:
                raise ValueError('user-defined face neighbours not handled')
            if z['ztype'] == 0:
                z['dims'] = [r.i32(), r.i32(), r.i32()]
            else:
                if z['ztype'] in (6, 7):
                    raise ValueError('polygon/polyhedron zones not handled')
                z['npts'] = r.i32()
                z['nel'] = r.i32()
                z['cdims'] = [r.i32(), r.i32(), r.i32()]
            # aux data: repeated (1, name, fmt, value), terminated by 0
            aux = []
            while True:
                flag = r.i32()
                if flag == 0:
                    break
                if flag != 1:
                    raise ValueError('unexpected aux flag {}'.format(flag))
                name = r.string(); fmt = r.i32(); value = r.string()
                aux.append((name, fmt, value))
            z['aux'] = aux
            aux_all.extend(('ZONE', n, v) for n, _, v in aux)
            zones.append(z)
        elif abs(marker - DS_AUX_MARKER) < 1e-3:
            name = r.string(); fmt = r.i32(); value = r.string()
            aux_all.append(('DATASET', name, value))
        elif abs(marker - VAR_AUX_MARKER) < 1e-3:
            ivar = r.i32(); name = r.string(); fmt = r.i32(); value = r.string()
            aux_all.append(('VAR{}'.format(ivar), name, value))
        elif abs(marker - EOH_MARKER) < 1e-3:
            eoh = f.tell() - 4
            break
        else:
            raise ValueError('unhandled header record marker {} at {}'.format(marker, f.tell() - 4))
    f.close()
    return {'byte_order': byte_order, 'file_type': file_type, 'title': title,
            'names': names, 'zones': zones, 'aux': aux_all, 'eoh': eoh}


def build_header(h, zone_name='global_field'):
    out = [b'#!TDV112', struct.pack('<ii', h['byte_order'], h['file_type']),
           enc_string(h['title']), struct.pack('<i', len(h['names']))]
    out += [enc_string(RENAME.get(n, n)) for n in h['names']]
    for z in h['zones']:
        out.append(struct.pack('<f', ZONE_MARKER))
        out.append(enc_string(zone_name if len(h['zones']) == 1 else z['name']))
        out.append(struct.pack('<iidii', z['parent'], z['strand'], z['soltime'],
                               z['unused'], z['ztype']))
        out.append(struct.pack('<i', z['varloc']))
        out += [struct.pack('<i', v) for v in z['varloc_list']]
        out.append(struct.pack('<ii', z['rawface'], z['misc']))
        if z['ztype'] == 0:
            out.append(struct.pack('<iii', *z['dims']))
        else:
            out.append(struct.pack('<ii', z['npts'], z['nel']))
            out.append(struct.pack('<iii', *z['cdims']))
        out.append(struct.pack('<i', 0))          # no aux data
    out.append(struct.pack('<f', EOH_MARKER))
    return b''.join(out)


def convert(src, dst, aux_path=None):
    h = parse_header(src)
    header = build_header(h)
    with open(src, 'rb') as fin, open(dst, 'wb') as fout:
        fout.write(header)
        fin.seek(h['eoh'] + 4)
        shutil.copyfileobj(fin, fout, 64 * 1024 * 1024)
    if aux_path:
        with open(aux_path, 'w') as fa:
            for scope, name, value in h['aux']:
                fa.write('{}:{}\n'.format(name, value) if scope == 'ZONE'
                         else '{} {}:{}\n'.format(scope, name, value))
    return h


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('paths', nargs='+')
    p.add_argument('--in-place', action='store_true',
                   help='rewrite each file in place, moving the original to preplot_orig/')
    p.add_argument('--orig-dir', default='preplot_orig')
    args = p.parse_args()

    if not args.in_place:
        if len(args.paths) != 2:
            sys.exit('usage: plt_visit_fix.py IN.plt OUT.plt  |  --in-place FILES...')
        src, dst = args.paths
        h = convert(src, dst, os.path.splitext(dst)[0] + '.aux')
        print('{} -> {}: zone {!r} -> global_field, {} aux records dropped'.format(
            src, dst, h['zones'][0]['name'], len(h['aux'])))
        return

    n = 0
    for src in args.paths:
        d = os.path.dirname(os.path.abspath(src))
        orig_dir = os.path.join(d, args.orig_dir)
        os.makedirs(orig_dir, exist_ok=True)
        base = os.path.basename(src)
        h = parse_header(src)
        if h['zones'] and h['zones'][0]['name'] == 'global_field' and not h['aux']:
            continue                                   # already converted (resumable)
        tmp = os.path.join(d, '.' + base + '.visitfix')
        convert(src, tmp, os.path.join(d, os.path.splitext(base)[0] + '.aux'))
        os.replace(src, os.path.join(orig_dir, base))
        os.replace(tmp, src)
        n += 1
        if n % 20 == 0:
            print('  {} converted'.format(n)); sys.stdout.flush()
    print('{} files converted in place; originals in {}'.format(n, args.orig_dir))


if __name__ == '__main__':
    main()
