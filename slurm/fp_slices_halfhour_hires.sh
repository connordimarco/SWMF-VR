#!/bin/bash
#SBATCH --job-name=swmfvr_fpsl_hh
#SBATCH --array=0-56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# High-resolution flux slices for the half-hour frames only (Mike Makivic,
# 2026-10-05): step 0.025 Re, 48 azimuth cuts (RBE has 48 MLT sectors), one
# frame per array task, written under their own OUT_ROOT so the 1-min 0.1 Re
# set in run/fp_slices is untouched. The slicer writes the sparse >2 MeV CSV
# itself (--bands gt2 --drop-zeros), so no re-export is needed; run
# scripts/fp_slices_csv.py once afterwards only to produce grid.json.
# Frame list: run/halfhour_frames.txt (frame nearest HH:15 / HH:45, 57 lines).
# Submit:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpslhh_%A_%a.out" \
#          slurm/fp_slices_halfhour_hires.sh
# Knobs (env): STEP (0.025), NSLICES (48), EXTENT (10), OUT_ROOT. Resumable.
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
STEP="${STEP:-0.025}"
NSLICES="${NSLICES:-48}"
EXTENT="${EXTENT:-10}"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run/fp_slices_hires_halfhour}"
FRAMES="${FRAMES:-$SWMF_VR_ROOT/run/halfhour_frames.txt}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$SWMF_VR_ROOT/run/logs" "$OUT_ROOT"

TS=$(sed -n "$((RANK + 1))p" "$FRAMES")
[[ -z "$TS" ]] && { echo "no frame for rank $RANK"; exit 0; }
[[ -f "$OUT_ROOT/fp_slices/npz/slices_${TS}.npz" ]] && { echo "frame $TS already done"; exit 0; }

echo "fp slices hires rank ${RANK} frame ${TS} step=${STEP} nslices=${NSLICES} extent=${EXTENT} -> ${OUT_ROOT}"
t0=$(date +%s)
"$PV_BATCH" "$SWMF_VR_ROOT/scripts/fp_slices.py" --ts "$TS" --no-plot \
    --step "$STEP" --extent "$EXTENT" --n-slices "$NSLICES" --frame gei \
    --bands gt2 --drop-zeros --out-root "$OUT_ROOT" 2>&1 \
    | grep -vE 'IGRF|^Warning|vtk|Generic Warning' || echo "frame $TS failed"
echo "frame $TS finished in $(( $(date +%s) - t0 )) s"
