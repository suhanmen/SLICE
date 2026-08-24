"""Screening: drop assertions that reject an input the description itself calls valid.

The inputs of the in-prompt examples are known-valid, so calling the assertion-carrying function on
them must not raise AssertionError. When it does, the offending assertion is identified by dropping
one at a time and removed. Only rejection is observed — no expected value is compared, and no
reference solution or held-out test is executed.

    python code/utils/screen_assertions.py <inserted.json> <final.json>
"""
import json, sys, os, ast, re, io, contextlib, multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
def _load(name):
    """Read an artifacts/ file on first use, with a pointer to the script that rebuilds it."""
    import json as _json, os as _os
    p = _os.path.join("artifacts", name)
    if not _os.path.exists(p):
        raise FileNotFoundError(f"{p} is missing — run scripts/build_artifacts.sh to rebuild it")
    return _json.load(open(p, encoding="utf-8"))


_CHECKS = None


def _checks(tid):
    global _CHECKS
    if _CHECKS is None:
        _CHECKS = _load("example_signals.json")
    return _CHECKS.get(tid, {}).get("checks", [])
CALL_RE = re.compile(r"assert\s+(.+?)\s*==", re.S)


def example_calls(tid):
    calls = []
    for c in _checks(tid):
        m = CALL_RE.search(c["stmt"])
        if m:
            calls.append(m.group(1).strip())
        elif c["stmt"].startswith("assert "):           # no comparison: take the call itself
            body = c["stmt"][len("assert "):].strip()
            if re.match(r"^\w+\s*\(", body):
                calls.append(body)
    return list(dict.fromkeys(calls))


def rejects(code, call, timeout=6.0):
    """True only if the call raises AssertionError; any other outcome is False."""
    def worker(q):
        env = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, env)
                try:
                    eval(call, env)
                    q.put(False)
                except AssertionError:
                    q.put(True)
                except Exception:
                    q.put(False)
        except Exception:
            q.put(False)

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(); return False
    try:
        return q.get_nowait()
    except Exception:
        return False


def fn_of(tree):
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return fns[0] if fns else None


def screen(code, calls):
    """Remove the assertions that reject an example input, one at a time."""
    removed = []
    for _ in range(10):                          # safety bound
        bad = [c for c in calls if rejects(code, c)]
        if not bad:
            break
        try:
            tree = ast.parse(code)
        except SyntaxError:
            break
        fn = fn_of(tree)
        if fn is None:
            break
        asserts = [s for s in fn.body if isinstance(s, ast.Assert)]
        if not asserts:
            break
        killer = None
        for s in asserts:                        # find the culprit by dropping one at a time
            t2 = ast.parse(code); f2 = fn_of(t2)
            f2.body = [x for x in f2.body if getattr(x, "lineno", -1) != s.lineno]
            c2 = ast.unparse(t2)
            if not any(rejects(c2, c) for c in bad):
                killer = (s.lineno, ast.unparse(s)); code = c2
                break
        if killer is None:                       # no single assertion explains it -> drop the first
            s = asserts[0]
            t2 = ast.parse(code); f2 = fn_of(t2)
            f2.body = [x for x in f2.body if getattr(x, "lineno", -1) != s.lineno]
            killer = (s.lineno, ast.unparse(s)); code = ast.unparse(t2)
        removed.append(killer[1])
    return code, removed


def main(in_path, out_path):
    recs = json.load(open(in_path))
    out = []
    n_screened = 0; n_removed = 0
    for r in recs:
        tid = str(r["id"])
        calls = example_calls(tid)
        rec = dict(r)
        # `n_asserts` is only present in stage (iii) output; other files (baselines) are screened
        # too, and screen() is a no-op when the function carries no assertion.
        if calls and ("n_asserts" not in r or (r.get("n_asserts") or 0) > 0):
            code, removed = screen(r["code"], calls)
            if removed:
                n_screened += 1; n_removed += len(removed)
                rec.update(code=code, screened_out=removed)
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[screen] {len(out)} tasks | fired on {n_screened} tasks, removed {n_removed} "
          f"assertions -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
