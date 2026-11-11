#!/usr/bin/env python3
"""Rebuild results/agent_check.jsonl from the per-run lines printed by the full-run cell of 08_agent_check
(used when the Kaggle session ended before agent_check_outputs.zip was downloaded, 11 Nov 2026).

Input : a text file with the copied cell output, lines like
        [tracelens s0] fastapi_13207: resolved=False gold_file=True calls=23 trace_ok=1 149.2s
Output: results/agent_check.jsonl with the fields 12_agent_stats.py needs. Fields that the printed lines do
        not carry (patch_chars, n_patch_files, graph_mentions, error_responses, malformed_calls) are null.
Usage : python scripts/08b_rebuild_jsonl_from_log.py run_log.txt [--out results/agent_check.jsonl]
"""
import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^\[(\w+) s(\d+)\] (\S+): resolved=(True|False) gold_file=(True|False) calls=(\S+) "
                  r"trace_ok=(\d+) ([\d.]+)s ?(.*)$")
REPO = {"fastapi": "fastapi/fastapi", "rich": "Textualize/rich", "requests": "psf/requests", "httpx": "encode/httpx"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(a.out) if a.out else root / "results" / "agent_check.jsonl"
    rows, bad = {}, 0
    for raw in Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(raw.strip())
        if not m:
            if raw.strip().startswith("["):
                bad += 1
            continue
        cond, seed, iid, res, gold, calls, tok, secs, err = m.groups()
        rows[(iid, cond, int(seed))] = {
            "instance_id": iid, "repo": REPO.get(iid.split("_")[0], iid.split("_")[0]), "condition": cond,
            "seed": int(seed), "resolved": res == "True", "gold_file_touched": gold == "True",
            "tool_calls": None if calls == "None" else int(calls), "seconds": float(secs), "wall_seconds": float(secs),
            "trace_ok": int(tok), "patch_chars": None, "n_patch_files": None, "graph_mentions": None,
            "error_responses": None, "malformed_calls": None, "error": err.strip() or None,
            "source": "rebuilt_from_printed_log",
        }
    if not rows:
        sys.exit("no run lines found: paste the output of the FULL-RUN cell (lines starting with [condition sN])")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter((r["condition"], r["seed"]) for r in rows.values())
    print(f"wrote {out}: {len(rows)} runs; unparsed bracket lines: {bad}")
    for k in sorted(c):
        print(f"  {k[0]:20s} seed {k[1]}: {c[k]}")
    exc = [k for k, r in rows.items() if (r["error"] or "").startswith("EXC:")]
    print("runs ending in an exception:", len(exc), exc[:10])


if __name__ == "__main__":
    main()
