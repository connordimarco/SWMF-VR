"""SLURM worker sharding -- two idioms, deliberately kept distinct.

The pipeline parallelizes by PROCESS (8 workers per stage), not MPI: each worker
gets a disjoint subset of timesteps via one of two patterns. They are NOT
interchangeable:

  * shard_modulo  -- Stage A streams the NPZ sequentially and can't materialize
                     a list, so it tests `t_idx % size == rank` on the fly.
  * shard_slice   -- Stage B globs a sorted file list and slices `list[rank::size]`.

Process-level sharding is also what keeps geopack.recalc's global state safe
(see swmf_vr.coords): one process handles its timesteps in order.
"""


def shard_modulo(idx, rank, size):
    """True if this worker owns streaming index `idx`. (Stage A.)"""
    return idx % size == rank


def shard_slice(items, rank, size):
    """The subset of a materialized, sorted list this worker owns. (Stage B.)"""
    return items[rank::size]


def parse_leading_ints(argv, count=3):
    """Pull up to `count` leading positional integers (rank, size, max_files)
    off argv, ignoring any non-integer tokens (e.g. --flags) without consuming
    them. Returns (ints, remaining_argv).

    Mirrors the lenient parsing the v6 scripts used: positional ints first, then
    flags. Stops collecting at the first non-int so flag values aren't eaten.
    """
    ints = []
    remaining = list(argv)
    consumed = []
    for tok in argv:
        if len(ints) >= count:
            break
        try:
            ints.append(int(tok))
            consumed.append(tok)
        except (ValueError, TypeError):
            break
    remaining = remaining[len(consumed):]
    return ints, remaining
