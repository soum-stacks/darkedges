"""Runtime call-graph tracer. Stdlib only.

Records, for code under `root`:
  nodes  : (relpath, qualname, firstlineno) with call counts and first-seen order
  edges  : caller -> callee with kind in {direct, lifted, entry} and counts
           direct = immediate Python caller frame is repo code
           lifted = nearest repo ancestor reached through external (non-repo) frames
           entry  = no repo ancestor on this thread's shadow stack
  tests  : per-test node/edge sets (via begin_test/end_test)
Frames named <listcomp>/<lambda>/<genexpr>/... are transparent: they inherit the
enclosing identity and record no edge.
"""
import json
import os
import sys
import threading
import time

_SKIP_DIRS = ("site-packages", "dist-packages", ".git", ".tox", ".venv", "venv", "__pycache__")
_TRANSPARENT = ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>")


class _Entry:
    __slots__ = ("code", "ident", "repo")

    def __init__(self, code, ident, repo):
        self.code, self.ident, self.repo = code, ident, repo


class _TestRec:
    def __init__(self, nodeid):
        self.nodeid = nodeid
        self.nodes = set()
        self.edges = set()
        self.outcome = None
        self.fail_frames = []


class Tracer:
    def __init__(self, root, use_monitoring=True):
        self.root = os.path.realpath(root)
        self.use_monitoring = use_monitoring and hasattr(sys, "monitoring")
        self.nodes = {}          # ident -> [index, calls]
        self.node_list = []      # ident by index
        self.edges = {}          # (caller_idx|None, callee_idx, kind) -> count
        self.tests = {}
        self._cur = None
        self._cls = {}           # code -> None (external) | ident | "T" (transparent)
        self._stacks = {}        # thread id -> [_Entry]
        self._seen_local = set()
        self._tool = None
        self._t0 = 0.0
        self._lock = threading.Lock()

    # ---- classification -------------------------------------------------
    def _classify(self, code):
        fn = code.co_filename
        res = None
        if fn and not fn.startswith("<"):
            try:
                real = os.path.realpath(fn)
            except OSError:
                real = fn
            if real.startswith(self.root + os.sep):
                rel = os.path.relpath(real, self.root).replace(os.sep, "/")
                if not any(part in _SKIP_DIRS for part in rel.split("/")):
                    if code.co_name in _TRANSPARENT:
                        res = "T"
                    else:
                        res = (rel, getattr(code, "co_qualname", code.co_name), code.co_firstlineno)
        self._cls[code] = res
        return res

    def _is_repo(self, code):
        info = self._cls.get(code, 0)
        if info == 0:
            info = self._classify(code)
        return info is not None

    def _node(self, ident):
        n = self.nodes.get(ident)
        if n is None:
            n = [len(self.node_list), 0]
            self.nodes[ident] = n
            self.node_list.append(ident)
        n[1] += 1
        return n[0]

    # ---- core push/pop ----------------------------------------------------
    def _push(self, code, info, immediate_repo, record_edge):
        tid = threading.get_ident()
        st = self._stacks.setdefault(tid, [])
        if info == "T":
            ident = st[-1].ident if st else None
            st.append(_Entry(code, ident, True))
            return
        callee = self._node(info)
        if record_edge:
            if st:
                kind = "direct" if immediate_repo else "lifted"
                caller = self.nodes[st[-1].ident][0] if st[-1].ident is not None else None
            else:
                kind, caller = "entry", None
            if st and st[-1].ident is None:
                kind, caller = "entry", None
            key = (caller, callee, kind)
            self.edges[key] = self.edges.get(key, 0) + 1
            if self._cur is not None:
                self._cur.edges.add(key)
        if self._cur is not None:
            self._cur.nodes.add(callee)
        st.append(_Entry(code, info, True))

    def _pop(self, code):
        st = self._stacks.get(threading.get_ident())
        if not st:
            return
        if st[-1].code is code:
            st.pop()
            return
        for i in range(len(st) - 1, -1, -1):
            if st[i].code is code:
                del st[i:]
                return

    # ---- sys.monitoring callbacks ----------------------------------------
    def _on_start(self, code, off):
        info = self._cls.get(code, 0)
        if info == 0:
            info = self._classify(code)
        if info is None:
            return sys.monitoring.DISABLE
        if code not in self._seen_local:
            self._seen_local.add(code)
            E = sys.monitoring.events
            sys.monitoring.set_local_events(
                self._tool, code, E.PY_RETURN | E.PY_YIELD | E.PY_RESUME)
        back = sys._getframe(1).f_back
        immediate_repo = back is not None and self._is_repo(back.f_code)
        self._push(code, info, immediate_repo, record_edge=True)

    def _on_resume(self, code, off):
        info = self._cls.get(code)
        if info is not None:
            self._push(code, info, True, record_edge=False)

    def _on_return(self, code, off, retval):
        self._pop(code)

    def _on_yield(self, code, off, retval):
        self._pop(code)

    def _on_unwind(self, code, off, exc):
        if self._cls.get(code) is not None:
            self._pop(code)

    # ---- setprofile fallback ---------------------------------------------
    def _profile(self, frame, event, arg):
        code = frame.f_code
        if event == "call":
            info = self._cls.get(code, 0)
            if info == 0:
                info = self._classify(code)
            if info is None:
                return
            back = frame.f_back
            immediate_repo = back is not None and self._is_repo(back.f_code)
            self._push(code, info, immediate_repo, record_edge=True)
        elif event == "return":
            if self._cls.get(code) is not None:
                self._pop(code)

    # ---- lifecycle --------------------------------------------------------
    def start(self):
        self._t0 = time.time()
        if self.use_monitoring:
            mon = sys.monitoring
            for tid in (mon.PROFILER_ID, 3, 4):
                try:
                    mon.use_tool_id(tid, "tracelens")
                    self._tool = tid
                    break
                except ValueError:
                    continue
            if self._tool is None:
                self.use_monitoring = False
        if self.use_monitoring:
            mon = sys.monitoring
            E = mon.events
            mon.register_callback(self._tool, E.PY_START, self._on_start)
            mon.register_callback(self._tool, E.PY_RESUME, self._on_resume)
            mon.register_callback(self._tool, E.PY_RETURN, self._on_return)
            mon.register_callback(self._tool, E.PY_YIELD, self._on_yield)
            mon.register_callback(self._tool, E.PY_UNWIND, self._on_unwind)
            mon.set_events(self._tool, E.PY_START | E.PY_UNWIND)  # PY_UNWIND cannot be local
        else:
            sys.setprofile(self._profile)
            threading.setprofile(self._profile)
        return self

    def stop(self):
        if self.use_monitoring and self._tool is not None:
            mon = sys.monitoring
            for code in list(self._seen_local):
                try:
                    mon.set_local_events(self._tool, code, 0)
                except Exception:
                    pass
            mon.set_events(self._tool, 0)
            for ev in (mon.events.PY_START, mon.events.PY_RESUME, mon.events.PY_RETURN,
                       mon.events.PY_YIELD, mon.events.PY_UNWIND):
                mon.register_callback(self._tool, ev, None)
            mon.free_tool_id(self._tool)
            self._tool = None
        else:
            sys.setprofile(None)
            threading.setprofile(None)

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # ---- per-test boundaries ----------------------------------------------
    def begin_test(self, nodeid):
        self._cur = self.tests.setdefault(nodeid, _TestRec(nodeid))

    def end_test(self):
        self._cur = None

    def set_outcome(self, nodeid, outcome, fail_frames=None):
        rec = self.tests.setdefault(nodeid, _TestRec(nodeid))
        if rec.outcome in (None, "passed") or outcome == "failed":
            rec.outcome = outcome
        if fail_frames:
            rec.fail_frames = fail_frames

    # ---- output -----------------------------------------------------------
    def to_dict(self):
        nodes = []
        for i, ident in enumerate(self.node_list):
            rel, qn, ln = ident
            nodes.append({"i": i, "relpath": rel, "qualname": qn, "firstlineno": ln,
                          "calls": self.nodes[ident][1]})
        edges = [{"caller": c, "callee": e, "kind": k, "count": n}
                 for (c, e, k), n in sorted(self.edges.items(), key=lambda kv: (kv[0][0] is None, kv[0]))]
        tests = {tid: {"outcome": r.outcome, "nodes": sorted(r.nodes),
                       "edges": sorted([list(x) for x in r.edges], key=str),
                       "fail_frames": r.fail_frames}
                 for tid, r in self.tests.items()}
        return {"root": self.root, "python": sys.version.split()[0],
                "backend": "monitoring" if self.use_monitoring else "setprofile",
                "wall_seconds": round(time.time() - self._t0, 3),
                "nodes": nodes, "edges": edges, "tests": tests}

    def dump(self, path):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
