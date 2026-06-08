#!/bin/bash
#SBATCH --mail-user=cdimarco@umich.edu
#SBATCH --mail-type=END,FAIL
#
# Submit the two stages as a dependent chain:
#   1. Stage A (submit_seeds.sh)  -- system python3: flux plots + seed CSVs
#   2. Stage B (submit_render.sh) -- pvbatch: field-line OBJ shells + screenshots
# Stage B waits for Stage A (afterok). Output root: $OUT_ROOT (default run/).
#
# Usage: ./submit_pipeline.sh
set -e

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

echo "Submitting Stage A (seed CSVs)..."
SEEDS_JOB=$(sbatch --parsable "$SCRIPT_DIR/submit_seeds.sh")
echo "  Stage A Job ID: $SEEDS_JOB"

echo "Submitting Stage B (render, waits for Stage A)..."
RENDER_JOB=$(sbatch --parsable --dependency=afterok:$SEEDS_JOB "$SCRIPT_DIR/submit_render.sh")
echo "  Stage B Job ID: $RENDER_JOB"

echo ""
echo "Pipeline submitted:"
echo "  1. Stage A seeds:  Job $SEEDS_JOB"
echo "  2. Stage B render: Job $RENDER_JOB (after $SEEDS_JOB)"
echo "Monitor: squeue -u \$USER   Cancel: scancel $SEEDS_JOB $RENDER_JOB"
