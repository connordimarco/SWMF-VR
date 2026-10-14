#!/bin/bash
#SBATCH --job-name=swmfvr_fpslcsv
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# Re-export the slice CSVs from run/fp_slices/npz in a new layout
# (scripts/fp_slices_csv.py; plain python3, a few seconds per frame). Submit:
#   source slurm/env.sh
#   CSV_ARGS="--bands gt2 --drop-zeros" OUT_DIR="$SWMF_VR_ROOT/run/fp_slices/csv_gt2_sparse" \
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpslcsv_%A_%a.out" \
#          slurm/fp_slices_csv_array.sh
# Knobs (env): CSV_ARGS (extra flags), OUT_DIR (default run/fp_slices/csv). Resumable.
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/fp_slices/csv}"
CSV_ARGS="${CSV_ARGS:-}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-8}"
mkdir -p "$OUT_ROOT/logs"

python3 "$SWMF_VR_ROOT/scripts/fp_slices_csv.py" "$RANK" "$NSHARDS" \
    --out-root "$OUT_ROOT" --out-dir "$OUT_DIR" $CSV_ARGS
