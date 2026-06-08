"""ParaView-only layer -- runs ONLY under pvbatch (ParaView's bundled Python).

Submodules here import `paraview.simple` and so will ImportError under the
system python3. Keep this __init__ free of paraview imports so that merely having
the subpackage on sys.path doesn't break the pure layer; the import only happens
when a submodule (tracing / extract / preview / crawl) is explicitly imported.
"""
