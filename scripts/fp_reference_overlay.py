"""Reference image for the CRA alignment test: one meridional plane of the >2 MeV
omni flux slices (npz cube, SM dipole frame) with the inner/outer shell field lines
that lie in that plane drawn on top. Everything is taken from the delivered files
(npz cube == csv_gt2_sparse, obj_fp_*_gei rotated GEI->SM with sm_axes_gei_fp.csv)."""
import sys, csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Arc

RUN = '/home/cdimarco/CRA/SWMF-VR/run'
ts = sys.argv[1] if len(sys.argv) > 1 else '20240511_094100'
VMIN, VMAX = 1e3, 1e8

rows = [r for r in csv.reader(l for l in open(f'{RUN}/sm_axes_gei_fp.csv') if not l.startswith('#'))]
v = np.array([r for r in rows[1:] if r[0] == ts][0][2:], float)
R = np.vstack([v[0:3], v[3:6], v[6:9]])          # rows = SM axes in GEI; sm = R @ gei

d = np.load(f'{RUN}/fp_slices/npz/slices_{ts}.npz', allow_pickle=True)
cube, az_all, rho, z = d['cube'], d['azimuths_deg'], d['rho_re'], d['z_re']
band = 2  # >2 MeV

def load_lines(name):
    V, L = [], []
    for line in open(f'{RUN}/obj_fp_{name}_gei/fieldlines_{ts}.obj'):
        if line.startswith('v '): V.append(line.split()[1:4])
        elif line.startswith('l '): L.append([int(t.split('/')[0]) - 1 for t in line.split()[1:]])
    V = np.array(V, float) @ R.T                     # GEI -> SM
    return [V[idx] for idx in L]

lines = {n: load_lines(n) for n in ('outer', 'inner')}

def line_az(P):
    # azimuth of the line's NORTH foot point (vertex 0, r = 1 Re, SM), degrees in [0, 360)
    p = P[0]
    return np.degrees(np.arctan2(p[1], p[0])) % 360

def apex_az(P):
    p = P[np.argmax(np.linalg.norm(P, axis=1))]
    return np.degrees(np.arctan2(p[1], p[0])) % 360

def plane(ax, az_right, az_left, title):
    ir, il = int(np.where(az_all == az_right)[0][0]), int(np.where(az_all == az_left)[0][0])
    img = np.concatenate([cube[il, :, ::-1, band], cube[ir, :, 1:, band]], axis=1)  # left = mirrored
    x = np.concatenate([-rho[::-1], rho[1:]])
    import copy; cmap = copy.copy(plt.get_cmap('inferno')); cmap.set_under('#101015')
    pm = ax.pcolormesh(x, z, np.clip(img, VMIN * 0.999, VMAX), cmap=cmap,
                       norm=LogNorm(vmin=VMIN, vmax=VMAX), shading='auto')
    for name, col, lab in (('outer', 'white', 'outer shell line'), ('inner', 'cyan', 'inner shell line')):
        first = True
        for P in lines[name]:
            a = line_az(P)
            for target, sign in ((az_right, 1), (az_left, -1)):
                if abs((a - target + 180) % 360 - 180) < 1.0:
                    ax.plot(sign * np.hypot(P[:, 0], P[:, 1]), P[:, 2], color=col, lw=1.6,  # projected onto the plane
                            label=lab if first else None); first = False
    ax.add_patch(Circle((0, 0), 1.0, color='#3060c0', zorder=5))
    ax.add_patch(Circle((0, 0), 2.5, fill=False, ls='--', lw=0.8, color='0.7', zorder=6))
    ax.set_aspect('equal'); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel('rho  [Re]   (negative = azimuth %d, positive = azimuth %d)' % (az_left, az_right))
    ax.set_ylabel('z  [Re]  (SM / dipole axis)'); ax.set_title(title, fontsize=9)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
    return pm

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
pm = plane(axes[0], 0, 180, 'noon-midnight plane: right = azimuth 0 (noon, MLT 12), left = azimuth 180 (midnight, MLT 0)')
plane(axes[1], 90, 270, 'dawn-dusk plane: right = azimuth 90 (dusk, MLT 18), left = azimuth 270 (dawn, MLT 6)')
cb = fig.colorbar(pm, ax=axes, fraction=0.025, pad=0.02)
cb.set_label('>2 MeV omnidirectional flux  [cm^-2 s^-1], log scale 1e3 - 1e8')
fig.suptitle(f'SWMF-VR reference frame {ts} UT.  Flux: fp_slices npz/csv_gt2_sparse (band flux_gt2MeV), SM dipole frame, '
             f'0.1 Re.  Lines: obj_fp_outer_gei / obj_fp_inner_gei rotated GEI->SM with sm_axes_gei_fp.csv.  '
             'Blue disc = Earth (1 Re), dashed = 2.5 Re dipole/BATSRUS handoff.', fontsize=8.5)
out = f'{RUN}/fp_reference_{ts}.png'
fig.savefig(out, dpi=130, bbox_inches='tight'); print('wrote', out)

# numbers for the email: equatorial crossings and flux peak per half-plane
print('\nhalf-plane  inner-shell rho@z=0  outer-shell rho@z=0  flux peak rho@z=0 (value)')
for a in (0, 180, 90, 270):
    ia = int(np.where(az_all == a)[0][0]); prof = cube[ia, np.argmin(np.abs(z)), :, band]
    xs = {}
    for name in ('inner', 'outer'):
        for P in lines[name]:
            if abs((line_az(P) - a + 180) % 360 - 180) < 1.0:
                k = np.argmin(np.abs(P[:, 2])); xs[name] = np.hypot(P[k, 0], P[k, 1])
    ip = np.argmax(prof)
    print(f'{a:>5d}       {xs.get("inner", float("nan")):.2f}                  {xs.get("outer", float("nan")):.2f}                 {rho[ip]:.1f}  ({prof[ip]:.2g})')

print('\napex azimuth drift from foot-point azimuth (deg), outer shell, all 48 lines:')
dr=np.array([ (apex_az(P)-line_az(P)+180)%360-180 for P in lines['outer']])
print('  median |drift| %.1f, max |drift| %.1f' % (np.median(np.abs(dr)), np.abs(dr).max()))
for a in (0,180,90,270):
    for P in lines['outer']:
        if abs((line_az(P)-a+180)%360-180)<1.0:
            print('  foot az %3d: apex az %.1f, apex r %.2f Re, z at apex %.2f' % (a, apex_az(P), np.linalg.norm(P,axis=1).max(), P[np.argmax(np.linalg.norm(P,axis=1)),2]))
