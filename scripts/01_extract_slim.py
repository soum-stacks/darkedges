"""Run in a Kaggle notebook with the main competition attached. Builds a small private zip (graphs + tasks + README)
and reports empty (0-byte) graph files. Do NOT publish the zip (competition data, Rule 4b)."""
import glob, json, os, re, shutil, collections

SRC = next(p for p in glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))
SRC = os.path.dirname(SRC)
OUT = "/kaggle/working/slim"; os.makedirs(OUT, exist_ok=True)
tasks = [json.loads(l) for l in open(f"{SRC}/tasks.jsonl")]
by_commit = {}
for t in tasks:
    by_commit.setdefault((t["repo"].split("/")[1].lower(), t["base_commit"]), []).append(t["instance_id"])

empty, ok = collections.Counter(), collections.Counter()
for (short, sha), ids in by_commit.items():
    cands = glob.glob(f"{SRC}/graphs/*{sha}.json") + [f"{SRC}/graphs/{i}.json" for i in ids]
    good = [c for c in cands if os.path.exists(c) and os.path.getsize(c) > 100]
    repo = short
    if not good:
        empty[repo] += 1; continue
    ok[repo] += 1
    shutil.copy(good[0], f"{OUT}/{short}_{sha}.json")
for n in ("tasks.jsonl", "HARNESS_README.md"):
    if os.path.exists(f"{SRC}/{n}"): shutil.copy(f"{SRC}/{n}", OUT)
print("tasks:", len(tasks), "| unique commits:", len(by_commit))
print("graphs OK per repo:", dict(ok)); print("graphs EMPTY/missing per repo:", dict(empty))
shutil.make_archive("/kaggle/working/slim", "zip", OUT)
print("wrote /kaggle/working/slim.zip")
