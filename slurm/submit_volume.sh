#!/bin/bash
#SBATCH --job-name=swmfvr_vol1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# One 3D flux volume frame (the voxel CSV the VR team asked for). ~2.5x the
# points of a y=0 slice at the default step, hence the bigger memory box.
# account/partition from SBATCH_* (set by .env); chdir/output on the sbatch line:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/vol1_%j.out" \
#          slurm/submit_volume.sh
# Override via env: TS, STEP, EXTENT, QUANTITY.
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/volume_renders/csv" "$OUT_ROOT/volume_renders/npz" \
         "$OUT_ROOT/volume_renders/seeds"

TS="${TS:-20240510_210000}"
STEP="${STEP:-0.5}"
EXTENT="${EXTENT:-10}"
QUANTITY="${QUANTITY:-omni}"

echo "vol1: ts=$TS step=$STEP extent=$EXTENT quantity=$QUANTITY -> $OUT_ROOT"
"$PV_BATCH" "$SWMF_VR_ROOT/scripts/volume3d.py" --ts "$TS" \
    --step "$STEP" --extent "$EXTENT" --quantity "$QUANTITY" --out-root "$OUT_ROOT"
echo "vol1 done."
