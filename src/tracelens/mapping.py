"""Map traced (relpath, qualname) to released-graph node ids such as 'fastapi.routing.APIRouter.get'."""


def module_candidates(relpath):
    """'src/pkg/sub/mod.py' -> ['src.pkg.sub.mod', 'pkg.sub.mod', 'sub.mod', 'mod'] (also drops __init__)."""
    parts = relpath[:-3].split("/") if relpath.endswith(".py") else relpath.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts))] if parts else []


def to_node_id(relpath, qualname, node_ids):
    """Return the matching graph node id or None. node_ids: a set of ids.
    Nested functions (qualname containing '<locals>') are unmapped by design."""
    if "<locals>" in qualname:
        return None
    for mod in module_candidates(relpath):
        if qualname == "<module>":
            cand = mod
        else:
            cand = f"{mod}.{qualname}"
        if cand in node_ids:
            return cand
    return None


def mapping_coverage(trace, node_ids):
    """Fraction of traced functions (weighted by calls) that map to a graph node."""
    tot = ok = 0
    mapped = {}
    for n in trace["nodes"]:
        nid = to_node_id(n["relpath"], n["qualname"], node_ids)
        mapped[n["i"]] = nid
        tot += n["calls"]
        ok += n["calls"] if nid else 0
    return (ok / tot if tot else 0.0), mapped
