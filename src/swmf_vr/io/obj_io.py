"""OBJ-level I/O: coordinate rewrite that preserves everything but vertices.

Verbatim port of v5/transform_obj_coords.transform_obj. Only 'v x y z' lines are
modified; all other content -- comments, `vt`, `mtllib`/`usemtl`, `l v/vt`
connectivity -- passes through unchanged. This is what keeps the v6 per-vertex
color (vt + ramp) intact through the GSM->GEI transform.
"""


def transform_obj(in_path, out_path, transform_fn):
    """Read OBJ, apply transform_fn(x, y, z) to every 'v x y z' line, write out.

    transform_fn returns a 3-tuple; vertices are written at %.10f (matching v5).
    All non-vertex lines are copied byte-for-byte.
    """
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                xt, yt, zt = transform_fn(x, y, z)
                fout.write("v {:.10f} {:.10f} {:.10f}\n".format(xt, yt, zt))
            else:
                fout.write(line)
