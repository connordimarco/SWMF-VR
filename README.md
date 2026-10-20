# SWMF-VR

Turns an SWMF Geospace run (BATSRUS + ionosphere + RBE or CIMI) into the CRA VR
products: belt-boundary shells, a cutaway, and flux on meridional slices.

## Pipeline

1. SWMF run (`scripts/new_storm_run.py` sets one up; `qsub job.athfe` in the run dir).
2. Convert outputs: `plt_visit_fix.py` (Tecplot headers), `cimi_fls_to_npz.py` (flux npz).
3. Belt edges per MLT sector at a fixed flux level: `footpoint_seeds.py`.
4. Field lines from the foot points (dipole below the handoff, BATSRUS above): `footpoint_trace.py`.
5. Products: cutaway render/movie (`fp_cutaway.py`), cutaway OBJ (`fp_cutaway_obj.py`),
   GEI shells (`gei.py`), slices + delivery CSVs (`fp_slices.py`, `fp_slices_csv.py`),
   delivery folder (`make_delivery.py`).

Method and decisions: `docs/`.

## Setup

Copy `.env.example` to `.env`, set the run dir, frame-0 time, npz path, handoff
radius, `pvbatch`/`ffmpeg`/Python paths. `source slurm/env.sh` before anything.

## Run

- Athena (PBS): `pbs/`. In order: `fp_cutaway_movie.pbs`, `fp_slices_deliver.pbs`,
  `fp_deliverables.pbs`. Each header has its `qsub` line and `-v` knobs.
- Great Lakes (Slurm): `slurm/`. Same stages as job arrays; see each header.
- Slice resolution: `-v STEP=0.025,NSLICES=72` on `fp_slices_deliver.pbs` (default 0.1 Re, 24 cuts).

## Conventions

GSM scene, y = 0 noon-midnight cut; delivered geometry in GEI (matrices in
`gsm_transforms_fp.csv`, slice axes in `grid.json`). Flux = > 2 MeV omnidirectional
electrons, cm^-2 s^-1, closed lines only. Timestamps `YYYYMMDD_HHMMSS`.
