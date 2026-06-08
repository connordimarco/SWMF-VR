"""swmf_vr -- Closed Radiation belt Analysis pipeline (consolidated from CRA v1-v6).

Capabilities: defining a boundary, seeding points from the boundary, resampling
the time axis, rotating into GEI, drawing concentric shells (field-line tracing),
and shading.

Runtime split (IMPORTANT):
  * The pure layer (config, sharding, timeaxis, boundary, seeding, coords,
    shading, io.*, presmooth.*) imports under BOTH the system python3 (3.6.8)
    and ParaView pvbatch (3.10). It never imports `paraview`.
  * The ParaView layer (swmf_vr.paraview.*) imports `paraview.simple` and runs
    ONLY under pvbatch.

Keep this __init__ free of heavy or paraview imports so `import swmf_vr` stays
cheap and safe under system python3.
"""

__version__ = "0.1.0"
