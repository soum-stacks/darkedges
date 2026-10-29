"""CLI: python -m tracelens.run --root ROOT --out OUT.json -- pytest [args] | script.py [args] | -m module [args]"""
import argparse
import os
import runpy
import sys
import traceback

from .tracer import Tracer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--setprofile", action="store_true", help="force the sys.setprofile backend")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("missing command after --")
    root = os.path.realpath(a.root)
    sys.path.insert(0, os.getcwd())
    tr = Tracer(root, use_monitoring=not a.setprofile)
    exit_code, repro_frames = 0, []
    tr.start()
    try:
        if cmd[0] == "pytest" or cmd[:2] == ["-m", "pytest"]:
            import pytest
            from .pytest_plugin import TraceLensPlugin
            args = cmd[1:] if cmd[0] == "pytest" else cmd[2:]
            exit_code = int(pytest.main(args, plugins=[TraceLensPlugin(tr)]))
        elif cmd[0] == "-m":
            sys.argv = [cmd[1]] + cmd[2:]
            runpy.run_module(cmd[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = cmd
            runpy.run_path(cmd[0], run_name="__main__")
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:
        exit_code = 1
        for fs in traceback.extract_tb(e.__traceback__):
            p = os.path.realpath(fs.filename)
            if p.startswith(root + os.sep):
                repro_frames.append([os.path.relpath(p, root).replace(os.sep, "/"), fs.lineno, fs.name])
        sys.stderr.write("".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:])
    finally:
        tr.stop()
    d = tr.to_dict()
    d["exit_code"] = exit_code
    d["script_fail_frames"] = repro_frames[-8:]
    import json
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(d, f)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
