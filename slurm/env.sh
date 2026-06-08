#!/usr/bin/env bash
# Source this before running any swmf_vr script, under either runtime:
#   source slurm/env.sh
#   python3      "$SWMF_VR_ROOT/scripts/seeds.py" ...   # plain python3
#   "$PV_BATCH"  "$SWMF_VR_ROOT/scripts/trace.py" ... # pvbatch
#
# The package is pure Python with no compiled extensions, so a single PYTHONPATH
# makes `import swmf_vr` resolve under BOTH the system python3 and pvbatch -- no
# per-interpreter install, no drift.

# Repo root = parent of this slurm/ dir, resolved absolutely.
SWMF_VR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SWMF_VR_ROOT
export PYTHONPATH="$SWMF_VR_ROOT/src:${PYTHONPATH:-}"

# ParaView OSMesa pvbatch (offscreen rendering). Relocated to CRA/tools/.
export PV_BATCH="/nfs/turbo/coe-tuija/shared/connor_austin/CRA/tools/ParaView-5.12.1-osmesa-MPI-Linux-Python3.10-x86_64/bin/pvbatch"
