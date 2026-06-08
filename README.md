# SWMF-VR

Turns the radiation-belt sim into 3D field-line shells for the VR app. 

Given 2D equatorial cut of flux, we find the belt boundaries (inner / outer / a "danger"
peak surface), trace the field lines through the GM magnetic field, color them,
rotate them into GEI (the frame the VR meshes go out in), and can draw a quick plot.

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

## Layout

`src/swmf_vr/`: `data` (read/write), `belt` (boundaries + seeds), `frames` (GSM↔GEI),
`color` (shading + 2D plot), `render` (ParaView tracing — pvbatch only), `config`.
Old versions live in `../archive/` if you need the history.
