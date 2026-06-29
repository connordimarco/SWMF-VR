#!/bin/bash
#SBATCH --job-name=swmfvr_slice1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# One y=0 slice frame (a smoke test / spot check). account/partition from SBATCH_*
# (set by .env); chdir/output passed on the sbatch line, e.g.:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/slice1_%j.out" \
#          slurm/submit_slice.sh
# Override via env: TS, N, EXTENT, QUANTITY.
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/slice_renders/png" "$OUT_ROOT/slice_renders/npz" \
         "$OUT_ROOT/slice_renders/seeds"

TS="${TS:-20240510_210000}"
N="${N:-121}"
EXTENT="${EXTENT:-10}"
QUANTITY="${QUANTITY:-omni}"

echo "slice1: ts=$TS n=$N extent=$EXTENT quantity=$QUANTITY -> $OUT_ROOT"
"$PV_BATCH" "$SWMF_VR_ROOT/scripts/slice3d.py" --ts "$TS" \
    --n "$N" --extent "$EXTENT" --quantity "$QUANTITY" --out-root "$OUT_ROOT"
echo "slice1 done."
