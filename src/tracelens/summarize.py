"""Compact trace summary for agents. Usage: python summarize.py TRACE.json [max_chars=4500] [--skip-tests]"""
import json
import sys


def summarize(path, max_chars=4500, skip_tests=True):
    d = json.load(open(path))
    nodes = {n["i"]: n for n in d["nodes"]}
    isT = lambda n: skip_tests and ("test" in n["relpath"].split("/")[-1] or "/tests/" in "/" + n["relpath"])
    out = [f"exit_code={d.get('exit_code')} backend={d['backend']} funcs={len(nodes)} edges={len(d['edges'])}"]
    ff = d.get("script_fail_frames") or [f for t in d["tests"].values() for f in t.get("fail_frames", [])]
    if ff:
        out.append("FAILURE FRAMES (repo, innermost last):")
        out += [f"  {r}:{l} {n}" for r, l, n in ff[-6:]]
    out.append("FIRST-CALLED REPO FUNCTIONS (order of first call):")
    for n in sorted(nodes.values(), key=lambda x: x["i"]):
        if not isT(n):
            out.append(f"  {n['relpath']}:{n['firstlineno']} {n['qualname']} x{n['calls']}")
    lifted = [e for e in d["edges"] if e["kind"] == "lifted" and e["caller"] is not None]
    if lifted:
        out.append("CALLS THROUGH FRAMEWORK FRAMES (static graphs usually miss these):")
        for e in lifted[:15]:
            out.append(f"  {nodes[e['caller']]['qualname']} -> {nodes[e['callee']]['qualname']} x{e['count']}")
    text = "\n".join(out)
    return text if len(text) <= max_chars else text[: max_chars - 20] + "\n...[truncated]"


if __name__ == "__main__":
    print(summarize(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4500))
