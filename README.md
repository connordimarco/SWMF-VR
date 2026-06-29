# SWMF-VR

Turns the radiation-belt sim into 3D field-line shells for the VR app. Connor + Austin.

Given a frame of flux, it finds the belt boundaries (inner / outer / a "danger"
peak surface), traces the field lines through the GM magnetic field, colors them,
rotates them into GEI (the frame the VR meshes go out in), and can draw a quick plot.

## Setup

```bash
source slurm/env.sh   # sets PYTHONPATH and PV_BATCH
```
Tracing needs ParaView's `pvbatch` (it imports `swmf_vr.render`). Everything else is
plain `python3`. No install — `env.sh` just puts `src/` on the path for both.

## Run one frame

```bash
./run.sh              # the 21:00 frame, colored by flux
./run.sh 480 1 radius # one frame, colored by radius instead
```
Output lands in `run/`:
- `out_seeds_*/` — boundary seed points;  `flux_slices/` — 2D flux plots
- `obj_*/` — traced shells (GSM), with `ramp.png` + `material.mtl` + `colormap.json`
- `obj_*_gei/` — the same shells in GEI (what VR gets)
- `gsm_transforms.csv` — the GSM→GEI rotation matrix per frame
- `shell_renders/` — the GSM-vs-GEI plot

## Pieces (run individually if you want)

```bash
python3     scripts/seeds.py 0 8 --out-root run        # flux -> seeds + plots
"$PV_BATCH" scripts/trace.py 0 8 --out-root run        # seeds -> shaded objs
python3     scripts/gei.py --obj-dir run/obj_outer --out-dir run/obj_outer_gei --transforms run/gsm_transforms.csv
python3     scripts/plot.py --ts 20240510_210000
python3     scripts/resample.py --src run/obj_outer --dst run/obj_outer_halfhour
```
`0 8` = worker 0 of 8 (the frames split across workers). For the full 1683 frames use
the SLURM scripts in `slurm/` (`./submit_all.sh`) — one process is too slow for that.

## Shading

`--color-by` picks the scalar baked into each vertex (`vt`, sampled against `ramp.png`):
`eqflux` (each line by its equatorial flux — the default), `eqflux_peak`, `lshell`,
`bmag`, `radius`, `lat`. The flux scales go up to the measured peak (~15875), so the
brightest color is the real maximum. Tune with `--cmap / --vmin / --vmax / --log`.

## 3D flux (CIMI → field lines)

The flux NPZ is CIMI output: directional flux on the *equator* only, per energy and
equatorial pitch angle. `slice3d.py` lifts that into 3D. For each point it traces the
BATSRUS field line, reads CIMI's equatorial flux at the line's minimum-|B| point, and
maps it out along the line with the two adiabatic invariants — `μ` opens the pitch
angle as `B` rises (`sin α = sin α_eq·√(B/B_eq)`), Liouville keeps the directional
flux fixed. Summed over the local cone that's the omnidirectional flux, which pancakes
toward the equator. **Only works on closed lines** (open ones have no trapped
equatorial reference); they're masked. Energy comes out in three bands (`<1`, `1-2`,
`>2 MeV`) integrated at the equator before mapping.

It samples a 2D plane cut (default `y=0`, the noon-midnight meridian) — to *see* the
mapping, and to make a movie of the storm. It's heavy (596 MB `.plt` + ~14k traces per
frame), so run it on a compute node, not the login node.

One frame:
```bash
sbatch slurm/submit_slice.sh                       # the 21:00 frame, y=0, omni
TS=20240510_210000 N=121 sbatch slurm/submit_slice.sh   # override via env
```
Each frame also carries **Bz / AL / Dst strips** below the maps (with a red line at
that frame's time) — parsed from the SWMF run logs. Build them once first:
```bash
python3 scripts/parse_indices.py                   # -> run/slice_renders/indices.npz
```
(`indices.npz` is picked up automatically if present; skip it and you just get the
maps.) Then all 1683 frames + stitch the movie (8 workers, frames sharded; re-runnable
— frames whose png+npz already exist are skipped, so a resubmit resumes):
```bash
sbatch slurm/submit_slice_all.sh                   # -> run/slice_renders/movie.mp4
```
Output in `run/slice_renders/`:
- `npz/slice_y0_<ts>_bands.npz` — the **raw** mapped flux (3 bands), unfiltered
- `png/slice_y0_<ts>_bands_omni.png` — the 3-panel heatmap per frame
- `movie.mp4` — the frames stitched (ffmpeg via `module load ffmpeg`; a `.jpg` can't
  hold a video)

The color scale is fixed per band (`sliceplot.LIMITS`, set just above the run's global
equatorial-omni peak) so frames don't flicker — that's what makes the movie watchable.

The trace saves raw flux, so the despeckle + plot re-tunes offline (plain `python3`, no
re-tracing) — and writes to `png/`:
```bash
python3 scripts/replot_slice.py --npz run/slice_renders/npz/slice_y0_<ts>_bands.npz
```
A few traces land a bad min-|B| and the flux collapses to ~`1e-46` (salt-and-pepper).
`despeckle` replaces *only* those — pixels far below their local (real-valued)
neighbour median, plus embedded NaN holes — and leaves every good pixel and the real
empty regions alone. Knobs: `--drop` (decades below local median to count as bad) and
`--floor`.

## Layout

`src/swmf_vr/`: `data` (read/write), `belt` (boundaries + seeds), `frames` (GSM↔GEI),
`color` (shell shading + 2D plot), `render` (ParaView tracing — pvbatch only),
`mapping` (the CIMI→3D adiabatic kernel + equatorial lookup), `sliceplot` (despeckle +
slice plot with the Bz/AL/Dst strips), `indices` (SWMF-log readers), `config`. The
3D-flux scripts are `slice3d.py` (trace, pvbatch), `replot_slice.py` (offline replot),
and `parse_indices.py` (build the index strips). Old versions live in `../archive/` if
you need the history.
