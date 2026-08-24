"""Lenient scoring: an intended rejection counts as `assert` **or** an explicit `raise`.

Classification is traceback-based. If the deepest frame of the exception raised by a
contract-violating call lies inside the generated code and that line is a `raise`, the rejection is
credited (`explicit_raise`); an AssertionError from an `assert` is `assertion`; anything else is an
incidental crash and is not credited. The five-way breakdown is otherwise identical to the strict
one -- only the cvt_miss test differs.
"""
import json, sys, os, io, re, ast, traceback, contextlib, hashlib, multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "code")
from utils.contract_eval import _score_sample, strip_checks, _load_excluded_checks, extract_python_code
from utils.build_expected_outputs import load_or_build_expected

_CACHE = None


def expected_checks():
    """Expected outputs per task, built on first use (the cache file is large and slow to build)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_or_build_expected("contracteval", ".", source="both")
    return _CACHE
EXC = _load_excluded_checks("contracteval", ".")
TASKS = {str(t["id"]): t for t in json.load(open("dataset/contracteval/test.json"))}


def _worker(code, calls, q):
    lines = code.splitlines()
    env = {}
    modes = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<cvt>", "exec"), env)
    except Exception:
        q.put(["error"] * len(calls)); return
    for call in calls:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                eval(compile(call, "<cvtcall>", "eval"), env)
            modes.append("ok")
        except AssertionError:
            modes.append("assertion")
        except Exception:
            tb = sys.exc_info()[2]
            mode = "error"
            deepest = None
            for fs in traceback.extract_tb(tb):
                if fs.filename == "<cvt>":
                    deepest = fs
            if deepest and 1 <= deepest.lineno <= len(lines):
                src = lines[deepest.lineno - 1].strip()
                if src.startswith("raise"):
                    mode = "explicit_raise"
            modes.append(mode)
    q.put(modes)


def cvt_modes(code, calls, timeout=15.0):
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(code, calls, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return ["error"] * len(calls)
    try:
        return q.get_nowait()
    except Exception:
        return ["error"] * len(calls)


def code_of(r):
    c = r.get("code") or ""
    return c if "def " in c else extract_python_code(c)


CACHE_DIR = "evaluation/scoring/pertask"


def _cache_path(path):
    """Cache keyed on the content hash of the output file."""
    h = hashlib.md5(open(path, "rb").read()).hexdigest()[:12]
    tag = path.replace("/", "__").replace(".json", "")
    return os.path.join(CACHE_DIR, f"{tag}__{h}.json")


def score(path, use_cache=True):
    """Ad-hoc console scoring; results_store.py is the persistent store."""
    raw = json.load(open(path))
    recs = raw if isinstance(raw, dict) else {str(r.get("id")): r for r in raw}
    n = 0
    c = {"strict": 0, "fw": 0, "or": 0, "cm": 0, "bw": 0}
    mode_tot = {}
    per_task = {}
    for tid, t in TASKS.items():
        r = recs.get(tid)
        ch = [[x["call"], x["expected_repr"], x["source"]] for x in expected_checks().get(tid, {}).get("checks", [])]
        d = EXC.get(tid)
        if d:
            ch = [x for i, x in enumerate(ch) if i not in d]
        if not ch or r is None:
            continue
        priv = t.get("private_cvts") or []
        n += 1
        code = code_of(r)
        full = _score_sample(code, ch, [], 15.0)
        body = _score_sample(strip_checks(code), ch, [], 15.0)
        ffl = list(full.get("func_base", [])) + list(full.get("func_plus", []))
        bfl = list(body.get("func_base", [])) + list(body.get("func_plus", []))
        over = any((not ffl[i]) and (i < len(bfl) and bfl[i]) for i in range(len(ffl)))
        modes = cvt_modes(code, [p if isinstance(p, str) else p[0] for p in priv]) if priv else []
        for m in modes:
            mode_tot[m] = mode_tot.get(m, 0) + 1
        miss = not (len(priv) == 0 or (len(modes) > 0 and all(m in ("assertion", "explicit_raise") for m in modes)))
        F = bool(body.get("loaded")) and bool(bfl) and all(bfl)
        if F and not over and not miss: lab = "strict"
        elif (not F) and not over and not miss: lab = "fw"
        elif F and over: lab = "or"
        elif F and miss and not over: lab = "cm"
        else: lab = "bw"
        c[lab] += 1
        per_task[tid] = {"bucket": lab, "func": bool(F), "over_rej": bool(over),
                         "cvt_miss": bool(miss), "cvt_modes": modes}
    P = lambda k: c[k] / n * 100
    print(f"{path}")
    print(f"  [lenient] SSR {P('strict'):5.2f}%  fw {P('fw'):5.2f}  or {P('or'):4.2f}  "
          f"cm {P('cm'):5.2f}  bw {P('bw'):4.2f}  (n={n})")
    tot = sum(mode_tot.values()) or 1
    print("  rejection modes:", {k: f"{v} ({v/tot*100:.1f}%)"
                                 for k, v in sorted(mode_tot.items(), key=lambda x: -x[1])})
    return {"path": path, "n": n, "counts": c, "mode_tot": mode_tot, "per_task": per_task}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-cache"]
    uc = "--no-cache" not in sys.argv
    for p in args:
        score(p, use_cache=uc)
