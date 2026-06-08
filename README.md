# SWMF-VR

Consolidated **Closed Radiation belt Analysis** pipeline: extract radiation-belt
boundaries from SWMF/GM simulation flux, trace them as 3D field-line surfaces,
and deliver colored OBJ meshes + coordinate metadata to a VR headset application.

This package replaces the scattered `CRA/v1`–`v6` script trees. Each of the six
core capabilities is now a single reusable function set, with the version-to-
version copy-paste removed. Default parameters reproduce the v6 campaign exactly.

## The six capabilities

| Capability | Module |
|---|---|
| Defining a boundary | `swmf_vr.boundary` |
| Seeding points from the boundary | `swmf_vr.seeding` |
| Resampling the time axis | `swmf_vr.timeaxis` (`resample_to_halfhour`) |
| Rotating into GEI | `swmf_vr.coords` |
| Drawing concentric shells (field-line tracing) | `swmf_vr.paraview.tracing` |
| Shading | `swmf_vr.shading` (+ `paraview.extract`, `paraview.preview`) |

Supporting: `swmf_vr.io` (NPZ streaming, seed CSV, OBJ rewrite), `swmf_vr.config`
(parameters), `swmf_vr.sharding`, `swmf_vr.plots` (2D diagnostic heatmaps).

## Dual runtime (important)

Two Python interpreters are involved and the package imports under both
**without an install step**:

- **System `python3` (3.6.8 here)** — boundary, seeding, coords, time, shading
  math, I/O. Deps: numpy, scipy, geopack, matplotlib.
- **ParaView `pvbatch` (bundled Python 3.10)** — field-line tracing + the
  screenshot preview, under `swmf_vr.paraview.*`.

The pure layer **never imports `paraview`**; only `swmf_vr.paraview.*` does. A
single `PYTHONPATH=src` makes `import swmf_vr` resolve under both. ParaView is
**not** a pip dependency — it is the runtime.

```bash
source slurm/env.sh    # sets PYTHONPATH=$REPO/src and PV_BATCH
```

## Running

Full pipeline via SLURM (Stage A seeds -> Stage B render, dependent chain):

```bash
cd slurm && ./submit_pipeline.sh      # outputs under $OUT_ROOT (default run/)
```

Individual stages (sharded `<rank> <size>`):

```bash
source slurm/env.sh
python3     scripts/stage_a_flux_seeds.py   0 8 --out-root run    # system python3
"$PV_BATCH" scripts/stage_b_render_lines.py 0 8 --out-root run    # pvbatch

# GEI export + transform, half-hour resampling (system python3):
python3 scripts/export_transforms.py --obj-dir run/obj_outer --output gsm_transforms.csv
python3 scripts/transform_obj.py    --target gei --obj-dir run/obj_outer --out-dir run/obj_outer_gei --copy-assets
python3 scripts/resample_halfhour.py --src run/obj_outer --dst run/obj_outer_halfhour
```

Stage B coloring is configurable: `--color-by {eqflux,eqflux_peak,lshell,bmag,radius,lat}`
`--cmap NAME --vmin V --vmax V --log/--no-log`. The OBJ carries the scalar in
`vt`; `ramp.png` + `material.mtl` + `colormap.json` are written alongside.

## Parity / tests

`tests/parity/run_parity.py` (system python3) proves the pure layer reproduces
v5/v6 byte-for-byte (seed CSVs, GSM->GEI transforms). `tests/parity/pvbatch_parity.sh`
proves Stage B's colored OBJ geometry + assets match v6 under pvbatch.

```bash
source slurm/env.sh
python3 tests/parity/run_parity.py
bash    tests/parity/pvbatch_parity.sh
```

## Coordinate frames

GSM (simulation native) → GEI/EME2000 (what collaborators consume) via geopack;
see `swmf_vr.coords`. `geopack.recalc(ut)` is global state — set once per
timestep before any transform; never interleave timesteps in one process (the
8-worker sharding is process-level, which is safe).
