# SWMF-VR

Turns the radiation-belt sim into 3D field-line shells for the VR app.

Given a 2D equatorial cut of flux, we find the belt boundaries (inner / outer / a
"danger" peak surface), trace the field lines through the GM magnetic field, color them,
rotate them into GEI (the frame the VR meshes go out in), and can draw a quick plot.

## Setup

Copy the config template, edit it for your machine (data paths, ParaView `pvbatch`,
SLURM account/partition), then source it:
```bash
cp .env.example .env   # then edit .env
source slurm/env.sh    # loads .env, sets PYTHONPATH + PV_BATCH (+ SBATCH_* for sbatch)
```
`.env` is the one place anything machine-specific lives — the Python (`config.py`) and
every SLURM script read from it (account/partition flow to `sbatch` via `SBATCH_*`, so
the job scripts carry no hardcoded paths). `.env` is gitignored; `env.sh` falls back to
`.env.example` (then to built-in literals) if you haven't made one. Tracing needs
ParaView's `pvbatch` (it imports `swmf_vr.render`); everything else is plain `python3`.
No install.

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

## 3D flux (RBE → field lines)

The flux NPZ is RBE output: directional flux at the *equator* only (the minimum-|B|
point of each line, not z=0), per energy and equatorial pitch angle. `slice3d.py` lifts that into 3D. For each point it traces the
BATSRUS field line, reads RBE's equatorial flux at the line's minimum-|B| point, and
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
(`submit_slice_all.sh` builds `indices.npz` for you if it's missing.) Then all 1683
frames + stitch the movie — a **job array** of 32 small ~13 GB tasks that backfill into
busy nodes instantly (one big node would wait), plus a dependent movie job. Re-runnable:
frames whose png+npz already exist are skipped, so a resubmit resumes.
```bash
./slurm/submit_slice_all.sh                        # -> run/slice_renders/movie.mp4
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

## Foot-point method (current, 2026-09)

The pipeline above seeds at the equator; the **current** method seeds from the
ionospheric foot points instead (Gabor's review, 2026-09-01 — see
`docs/2026-09-01_footpoint_redesign.md` for the why, the decisions and the numbers).
Each line is the tilted centered dipole from the foot point up to 2.5 Re, the BATSRUS
trace beyond, and the dipole back down from where the trace re-enters; the flux
volume comes from BATSRUS's own foot-point arrays with no tracing at all. Same OBJ /
CSV formats, so `gei.py`, `volume_gei.py` and the VR app are unchanged.

```bash
python3     scripts/footpoint_seeds.py 0 8 --out-root run     # boundaries in foot-point latitude -> out_seeds_fp_*/
"$PV_BATCH" scripts/footpoint_trace.py 0 8 --out-root run     # hybrid dipole+BATSRUS lines -> obj_fp_*/
"$PV_BATCH" scripts/footpoint_volume.py --ts 20240510_210000  # the flux box, ~10 s/frame -> volume_renders_fp/
python3     scripts/fp_movie.py 0 8                            # field-line movie frames
python3     scripts/fp_volume_cuts.py phases --ts <ts> ...     # three cuts through the box (or `movie`)
"$PV_BATCH" scripts/fp_cutaway.py --ts <ts>                    # half-donut cutaway render
"$PV_BATCH" scripts/fp_cutaway_obj.py --ts <ts> --frame gei    # the cutaway as OBJ for VR (+ PARAVIEW_INSTRUCTIONS.md)
```
Full run: `slurm/submit_fp_seeds.sh` → `slurm/fp_trace_array.sh` →
`slurm/fp_volume_array.sh` / `slurm/fp_cutaway_array.sh` (32-task arrays, resumable).
Knobs worth knowing: `--apex-max` (lines stretching beyond it, or open, retreat
equatorward until trapped; 12 Re), `--handoff` (2.5 Re), `OUTER_GAP_ROWS` in
`footpoint_seeds.py`.

## Meridional slices about the dipole axis (2026-09)

For the VR team's 3D reconstruction: 24 half-plane slices, 15 deg apart about the
SM (dipole) axis, flux mapped directly on each plane (no voxel resampling), all
three bands, **no NaNs** (empty = 0, interior pinholes filled). One frame, all
slices, ~90 s. See `docs/2026-09-08_dipole_slices.md` (incl. the |B| < B_min
finding on the flanks; the per-pixel ratio is saved as `bratio`).

```bash
"$PV_BATCH" scripts/fp_slices.py --ts 20240511_094100 [--step 0.1] [--frame gei|sm|gsm]   # -> run/fp_slices/{csv,npz}/ + preview png
python3     scripts/fp_slices_movie.py 0 24 [--bmin-tol 0.8]                            # one movie per azimuth -> run/fp_slices/movies/
python3     scripts/fp_slices_csv.py [--bands gt2] [--drop-zeros] [--split-azimuths] [--out-dir D]   # re-export CSV from the npz cubes (default: one file per ts, all cuts, azimuth_deg column)
```
Full run: `slurm/fp_slices_array.sh` (32 shards) then `slurm/fp_slices_movie_array.sh`
(24 tasks, `--dependency=afterany`). Azimuth 0 = noon, MLT = az/15 + 12.

## Athena (NASA HECC) and the Galaxy-15 (April 2010) CIMI run

The April 2010 run (`/nobackupp28/cdimarco/run_galaxy2010`, BATSRUS + RIM +
CIMI, 3D `tcp` files every 10 min, CIMI `fls e` every minute) is processed on
Athena. Differences from the Mother's Day / RBE setup, all driven by `.env`:

- `scripts/cimi_fls_to_npz.py` converts CIMI's `.fls` electron files to the
  RBE npz layout (75 lat x 48 MLT x 20 E x 18 pitch angles; MLT axis rolled to
  start at 0; `irm` recovered from the frozen rows; `time` = minutes after
  `SWMF_BASE_ISO`). `--plt-dir` keeps only the times that have a 3D `.plt`.
- `SWMF_BASE_ISO` (new) sets frame 0; the frame index is looked up on the npz
  time axis (`data.frame_index_for`), so the cadence no longer has to be 1 min.
- `footpoint_seeds.py` takes the grid size from the npz.
- `scripts/plt_visit_fix.py`: preplot's raw `.plt` output (zone "3D N=...",
  a chain of AUXDATA records, `X [R]` names) is rejected as invalid by
  ParaView's VisIt Tecplot reader in every build tried. Austin's
  `3d__paraview_1_*.plt` read fine because they were rewritten: zone
  `global_field`, no aux, `x y z`. The script does the same header rewrite
  (data section copied byte for byte; aux saved as `<name>.aux`, originals in
  `preplot_orig/`). The PBS job runs it first.
- `pbs/fp_cutaway_movie.pbs`: the whole chain on one Turin node (plt fix, npz,
  seeds, trace, cutaway, ffmpeg), `NPAR` pvbatch processes at a time.

Athena tools (see the LAUREN project notes for the login chain): ParaView
5.12.1 osmesa binary in `/nobackupp28/cdimarco/CRA/tools/` (plus `libglapi`
copied from NASA's 5.11 mesa build), static ffmpeg there too, `cray-python`
venv `/nobackupp28/cdimarco/CRA/venv` (matplotlib) and `pip install --user
geopack` with the IGRF tables copied in (the PyPI 1.0.13 package downloads
them at import, which compute nodes cannot do; `init_igrf` patched to skip the
fetch when tables exist).

## Layout

`src/swmf_vr/`: `data` (read/write), `belt` (boundaries + seeds), `frames` (GSM↔GEI),
`color` (shell shading + 2D plot), `render` (ParaView tracing — pvbatch only),
`mapping` (the RBE→3D adiabatic kernel + equatorial lookup), `sliceplot` (despeckle +
slice plot with the Bz/AL/Dst strips), `indices` (SWMF-log readers), `config`. The
3D-flux scripts are `slice3d.py` (trace, pvbatch), `replot_slice.py` (offline replot),
and `parse_indices.py` (build the index strips). Old versions live in `../archive/` if
you need the history.
