#!/bin/bash
#SBATCH --job-name=swmfvr_fpplanes
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# Full-plane slice movies with the shell mesh outline (scripts/fp_planes_movie.py).
# Tasks 0-15 render the noon-midnight plane, 16-31 the dawn-dusk plane, each a
# shard of frames (resumable). Afterwards stitch on a login node:
#   module load ffmpeg; source slurm/env.sh
#   python3 scripts/fp_planes_movie.py noon-midnight --stitch
#   python3 scripts/fp_planes_movie.py dawn-dusk --stitch
# Submit:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpplanes_%A_%a.out" slurm/fp_planes_array.sh
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
ID="${SLURM_ARRAY_TASK_ID:-0}"
PER_PLANE=16
PLANES=(noon-midnight dawn-dusk)
PLANE="${PLANES[$((ID / PER_PLANE))]}"
RANK=$((ID % PER_PLANE))
mkdir -p "$OUT_ROOT/logs"

python3 "$SWMF_VR_ROOT/scripts/fp_planes_movie.py" "$PLANE" "$RANK" "$PER_PLANE" --out-root "$OUT_ROOT"
