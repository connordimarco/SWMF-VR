# SWMF-VR

Turns an SWMF Geospace run (BATSRUS + ionosphere + an inner-magnetosphere model
with a radiation-belt population: RBE for the May 2024 run, CIMI for April 2010)
into the products the CRA VR team ingests: belt-boundary shells as field lines,
a half-donut cutaway, and flux on meridional slices they rebuild into a 3D texture.

## What happens, start to finish

1. **SWMF run** (Athena: `scripts/new_storm_run.py` sets up the run directory from a
   template, then `qsub job.athfe` in it). The run writes a 3D Tecplot file every
   10 min with the MHD state, open/closed status and the northern foot-point
   angles of every cell, and the belt model's electron flux on its foot-point grid.
2. **Convert the outputs.** `scripts/plt_visit_fix.py` rewrites the Tecplot headers
   so ParaView's VisIt reader accepts them; `scripts/cimi_fls_to_npz.py` turns the
   CIMI `.fls` files into the one npz the rest reads (RBE runs come as that npz).
3. **Belt edges** (`scripts/footpoint_seeds.py`): per frame and MLT sector, the
   inner edge, outer edge and peak of the > 2 MeV flux in foot-point latitude, at a
   fixed flux level, with an optional running median across sectors.
4. **Field lines** (`scripts/footpoint_trace.py`, pvbatch): each edge foot point
   becomes one line: the run's tilted dipole from the foot up to the handoff
   sphere, the BATSRUS trace beyond, the dipole back down. 48 lines per shell.
5. **Flux anywhere** (`src/swmf_vr/mapping.py` + `footpoint_volume.py`): a point's
   foot point comes from BATSRUS (dipole inside the handoff), the flux there is
   looked up on the belt grid and Liouville-mapped to the local |B|. This is what
   colours the cutaway and fills the slices.
6. **Products**: the cutaway render and movie (`fp_cutaway.py`), the cutaway as an
   OBJ package (`fp_cutaway_obj.py`), the shells rotated to GEI (`gei.py`), the 24
   meridional half-plane slices and their delivery CSVs (`fp_slices.py`,
   `fp_slices_csv.py`), and the delivery folder with README and manifest
   (`make_delivery.py`).

The method and its decisions are written up in `docs/`:
`2026-09-01_footpoint_redesign.md` (why foot points, the handoff, the edge rules)
and `2026-09-08_dipole_slices.md` (the slices and the CSV layout).

## Setup

Everything machine-specific lives in `.env` (gitignored; start from `.env.example`):
the run directory, the frame-0 time, the npz path, the dipole/BATSRUS handoff radius
(a few cells above the run's rBody), and the `pvbatch`, `ffmpeg` and Python to use.
`source slurm/env.sh` loads it and puts `src/` on the path. Python needs numpy, scipy,
matplotlib and geopack; tracing and sampling run under ParaView's `pvbatch` (an
osmesa build for headless nodes), which must see numpy and geopack too.

## Running it

- **Athena (NASA HECC, PBS):** see `pbs/`. Three jobs, run in order: 
  `fp_cutaway_movie.pbs` (convert, edges, lines, cutaway frames, movie),
  `fp_slices_deliver.pbs` (slices and delivery CSVs), `fp_deliverables.pbs`
  (GEI shells, OBJ packages, delivery folder). Each is one node, resumable, and
  takes its knobs from `-v` (`NPAR`, `OUT_ROOT`, `OBJ_TS`, the `*_ARGS`). The header
  comment of each file has the exact `qsub` line.
- **Great Lakes (Slurm):** see `slurm/`. The same stages as job arrays:
  `submit_fp_seeds.sh`, `fp_trace_array.sh`, `fp_cutaway_array.sh`,
  `fp_slices_array.sh`, `fp_slices_csv_array.sh`, plus the volume, planes and
  movie arrays. Each file's header says how to submit it.

One frame by hand, for checking, after `source slurm/env.sh`:

```bash
python3     scripts/footpoint_seeds.py 0 1 --out-root run --max-frames 1
"$PV_BATCH" scripts/footpoint_trace.py 0 1 --match <ts> --out-root run
"$PV_BATCH" scripts/fp_cutaway.py --ts <ts> --out-root run
"$PV_BATCH" scripts/fp_slices.py --ts <ts> --out-root run
```

## Conventions

- Scene and cut plane are GSM (+x to the Sun, +z north); the noon-midnight plane
  y = 0 is shared with SM. Delivered geometry is GEI, with the per-frame GSM to GEI
  matrices in `gsm_transforms_fp.csv`; slice positions are GEI with the SM axes in
  `grid.json`. Azimuth 0 = noon, increasing toward dusk.
- Flux is the > 2 MeV omnidirectional electron flux in cm^-2 s^-1 (log colour
  table, range set per run). Only closed field lines carry flux.
- Timestamps are `YYYYMMDD_HHMMSS`; frames are whatever 3D files the run wrote
  (1 min for May 2024, 10 min for April 2010), matched on the npz time axis.

## Layout

`src/swmf_vr/`: `config` (knobs, env-driven paths), `data` (npz streaming, seed CSVs,
OBJ I/O, time), `dipole` (the analytic tilted dipole and the handoff), `frames`
(GSM/GEO/GEI via geopack), `mapping` (the adiabatic flux kernel), `render`
(ParaView reader and tracer), `color` (OBJ shading assets), `slices_csv` (the
delivery CSV writer), `sliceplot` and `indices` (the older 2D slice movies with
index strips). `scripts/` holds the stages above plus the older equatorial pipeline
(`seeds.py`, `trace.py`, `slice3d.py`, `volume3d.py`) kept for comparison;
`pbs/` and `slurm/` the job files; `docs/` the write-ups.
