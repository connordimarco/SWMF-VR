"""Legacy pre-smoothing stages (v1/v3 only) -- intentionally NOT migrated.

The Chaikin corner-cutting (smooth_seeds_spline.py), 4x interpolation
(interpolate_seeds.py), and inward crawl (crawl_seeds_inward.py) belong to the
old v1/v3 pipeline. The v6 path -- which this package consolidates -- writes
seeds straight from the boundary and traces them directly, so none of these run.

They are deliberately left in archive/v1 rather than ported as dead code. If a
future change needs corner-cutting on the boundary curve, port chaikin_closed /
resample_closed_curve / interpolate_periodic from archive/v1/smooth_seeds_spline.py
here (pure numpy); crawl_seeds_inward is pvbatch-only (it probes the GM field).
"""
