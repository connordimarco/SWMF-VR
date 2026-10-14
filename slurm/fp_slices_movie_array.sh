#!/bin/bash
#SBATCH --job-name=swmfvr_fpslmov
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# One movie per azimuth slice, over the whole storm (scripts/fp_slices_movie.py).
# Task k renders azimuth k's frames (matplotlib, resumable) and stitches them
# with ffmpeg -> run/fp_slices/movies/slice_az<NNN>.mp4. Submit after the
# slices array (--dependency=afterany:<JOB>):
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpslmov_%A_%a.out" \
#          slurm/fp_slices_movie_array.sh
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-24}"
mkdir -p "$OUT_ROOT/logs"
module load ffmpeg 2>/dev/null || module load ffmpeg/6.0.0 2>/dev/null || true

# OUTLINES=1 draws the in-plane shell field lines (-> movie_frames_outlined/, movies_outlined/).
python3 "$SWMF_VR_ROOT/scripts/fp_slices_movie.py" "$RANK" "$NSHARDS" --out-root "$OUT_ROOT" \
    ${FPS:+--fps "$FPS"} ${OUTLINES:+--outlines}
