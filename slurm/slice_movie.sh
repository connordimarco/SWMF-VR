#!/bin/bash
#SBATCH --job-name=swmfvr_movie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --get-user-env

# Stitch the rendered frames into a movie (runs after the slice array finishes).
# account/partition from SBATCH_* (set by .env); chdir/output via the sbatch line.
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
SLICE="$OUT_ROOT/slice_renders"
QUANTITY="${QUANTITY:-omni}"
FPS="${FPS:-24}"

NFRAMES=$(ls "$SLICE/png"/slice_y0_*_bands_${QUANTITY}.png 2>/dev/null | wc -l)
echo "stitching $NFRAMES frames at ${FPS} fps  (a .jpg can't hold a video -> movie.mp4)"
module load ffmpeg 2>/dev/null || module load ffmpeg/6.0.0 2>/dev/null || true
if command -v ffmpeg >/dev/null 2>&1 && [[ "$NFRAMES" -gt 0 ]]; then
    ffmpeg -y -framerate "$FPS" -pattern_type glob \
        -i "$SLICE/png/slice_y0_*_bands_${QUANTITY}.png" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p \
        "$SLICE/movie.mp4" && echo "wrote $SLICE/movie.mp4" \
        || echo "ffmpeg failed; frames are in $SLICE/png"
else
    echo "no ffmpeg or no frames; skipped. Frames in $SLICE/png"
fi
