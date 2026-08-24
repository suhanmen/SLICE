"""Per-task result store: score a (setting, model) pair once, keep the result.

One pass computes everything: SSR under both the lenient and the strict rejection rule, functional
correctness of the full program and of the body alone, and the five-way breakdown. A store whose
source file is byte-identical is returned as is, so assembling a table re-scores nothing, and a
task whose code did not change is carried over from another store of the same model rather than
re-executed.

    PYTHONPATH=code python code/utils/results_store.py score SLICE --models Qwen3.5-9B
    PYTHONPATH=code python code/utils/results_store.py table SLICE base_cs

Writes evaluation/contracteval/<model>/<setting>__breakdown.json as
  {model, dataset, condition, scoring, source, src_hash, aggregate{...}, per_task[{id, ...}]}
"""
import json, sys, os, io, glob, argparse, hashlib, traceback, contextlib, multiprocessing as mp

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
STORE = "evaluation/contracteval"
SUFFIX = "__breakdown"                   # <model>/<setting>__breakdown.json

MODELS = ["Qwen3.5-9B", "gemma-4-12B-it", "Llama-3.1-8B-Instruct",
          "gpt-5.4-nano"]                # default sweep; override with --models

# setting -> source path template, {S} = model, {T} = setting. Every setting scores final.json
# of its own run directory; a template is only needed for anything that deviates.
SETTINGS = {
    "SLICE":    "output/{T}/{S}/final.json",       # the method, from scripts/run_slice.sh
    "base_cs":  "output/{T}/{S}/final.json",       # baseline: problem + contract prose
    "base_eas": "output/{T}/{S}/final.json",       # baseline: + contract-violating examples
}
OVERRIDE = {}   # per-model exceptions: (setting, model) -> path



def src_path(setting, model):
    return (OVERRIDE.get((setting, model))
            or SETTINGS.get(setting, "output/{T}/{S}/final.json").format(S=model, T=setting))


def _worker(code, calls, q):
    lines = code.splitlines(); env = {}; modes = []
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
            tb = sys.exc_info()[2]; mode = "error"; deepest = None
            for fs in traceback.extract_tb(tb):
                if fs.filename == "<cvt>":
                    deepest = fs
            if deepest and 1 <= deepest.lineno <= len(lines):
                if lines[deepest.lineno - 1].strip().startswith("raise"):
                    mode = "explicit_raise"
            modes.append(mode)
    q.put(modes)


def cvt_modes(code, calls, timeout=15.0):
    q = mp.Queue(); p = mp.Process(target=_worker, args=(code, calls, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(); return ["error"] * len(calls)
    try:
        return q.get_nowait()
    except Exception:
        return ["error"] * len(calls)


def code_of(r):
    c = r.get("code") or ""
    return c if "def " in c else extract_python_code(c)


def _reuse_map(model, skip=()):
    """Build (task id, code md5) -> stored result from this model's existing stores.
    Identical code is never re-executed, which also keeps timeout-boundary noise out."""
    m = {}
    for f in sorted(glob.glob(os.path.join(STORE, model, f"*{SUFFIX}.json"))):
        if os.path.basename(f) in skip:      # do not reuse the store being rebuilt
            continue
        try:
            d = json.load(open(f))
            src = d.get("source")
            if not src or not os.path.exists(src):
                continue
            if hashlib.md5(open(src, "rb").read()).hexdigest()[:12] != d.get("src_hash"):
                continue                      # source changed: the store no longer matches
            raw = json.load(open(src))
            recs = raw if isinstance(raw, dict) else {str(r.get("id")): r for r in raw}
            for v in d.get("per_task", []):
                r = recs.get(str(v["id"]))
                if r is None:
                    continue
                h = hashlib.md5((code_of(r) or "").encode()).hexdigest()[:12]
                m.setdefault((str(v["id"]), h), {k: v[k] for k in
                             ("lenient", "strict_mode", "func_full", "body_func", "over_rej", "cvt_modes")})
        except Exception:
            continue
    return m


def evaluate(path, reuse=None):
    """Score one output file, returning per_task and aggregate for both schemes.
    reuse: (id, code md5) -> stored result; a task with identical code is not re-scored."""
    raw = json.load(open(path))
    recs = raw if isinstance(raw, dict) else {str(r.get("id")): r for r in raw}
    per, n, n_reuse = {}, 0, 0
    for tid, t in TASKS.items():
        r = recs.get(tid)
        ch = [[x["call"], x["expected_repr"], x["source"]] for x in expected_checks().get(tid, {}).get("checks", [])]
        d = EXC.get(tid)
        if d:
            ch = [x for i, x in enumerate(ch) if i not in d]
        if not ch or r is None:
            continue
        n += 1
        code = code_of(r)
        if reuse:
            hit = reuse.get((tid, hashlib.md5((code or "").encode()).hexdigest()[:12]))
            if hit is not None:
                per[tid] = dict(hit); n_reuse += 1; continue
        full = _score_sample(code, ch, [], 15.0)
        body = _score_sample(strip_checks(code), ch, [], 15.0)
        ffl = list(full.get("func_base", [])) + list(full.get("func_plus", []))
        bfl = list(body.get("func_base", [])) + list(body.get("func_plus", []))
        over = any((not ffl[i]) and (i < len(bfl) and bfl[i]) for i in range(len(ffl)))
        priv = t.get("private_cvts") or []
        modes = cvt_modes(code, [p if isinstance(p, str) else p[0] for p in priv]) if priv else []
        ok_len = len(priv) == 0
        miss_len = not (ok_len or (modes and all(m == "assertion" for m in modes)))
        miss_shortcut = not (ok_len or (modes and
                             all(m in ("assertion", "explicit_raise") for m in modes)))
        F = bool(body.get("loaded")) and bool(bfl) and all(bfl)
        func_full = bool(full.get("loaded")) and bool(ffl) and all(ffl)
        def bucket(miss):
            if F and not over and not miss: return "strict"
            if (not F) and not over and not miss: return "fw"
            if F and over: return "or"
            if F and miss and not over: return "cm"
            return "bw"
        per[tid] = {"lenient": bucket(miss_shortcut), "strict_mode": bucket(miss_len),
                    "func_full": func_full, "body_func": bool(F), "over_rej": bool(over),
                    "cvt_modes": modes}
    agg = {}
    for scheme in ("lenient", "strict_mode"):
        c = {k: 0 for k in ("strict", "fw", "or", "cm", "bw")}
        for v in per.values():
            c[v[scheme]] += 1
        P = lambda k: round(c[k] / max(n, 1) * 100, 2)
        agg[scheme] = {k: P(k) for k in c}
        agg[scheme]["csr"] = round(100 - agg[scheme]["or"] - agg[scheme]["cm"] - agg[scheme]["bw"], 2)
        agg[scheme]["strict_fail"] = round(100 - agg[scheme]["strict"], 2)
    agg["full_code_func"] = round(sum(v["func_full"] for v in per.values()) / max(n, 1) * 100, 2)
    agg["body_func"] = round(sum(v["body_func"] for v in per.values()) / max(n, 1) * 100, 2)
    agg["n_tasks"] = n
    agg["n_reused"] = n_reuse
    return per, agg


def do_score(settings, models, force=False, no_reuse=False):
    reuse_cache = {}
    for st in settings:
        for M in models:
            p = src_path(st, M)
            if not os.path.exists(p):
                print(f"[skip] {st}/{M}: no source ({p})"); continue
            h = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
            out = os.path.join(STORE, M, f"{st}{SUFFIX}.json")
            if not force and os.path.exists(out) and json.load(open(out)).get("src_hash") == h:
                print(f"[cached] {st}/{M}"); continue
            if no_reuse:
                rmap = None
            elif force:
                rmap = _reuse_map(M, skip=(f"{st}{SUFFIX}.json",))
            else:
                rmap = reuse_cache.setdefault(M, _reuse_map(M))
            per, agg = evaluate(p, rmap)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump({"model": M, "dataset": "contracteval", "condition": st,
                       "scoring": "decompose5 (lenient + strict)", "source": p, "src_hash": h,
                       "aggregate": agg,
                       "per_task": [dict(id=tid, **v) for tid, v in sorted(per.items())]},
                      open(out, "w"), indent=1)
            print(f"[scored] {st}/{M}: lenient SSR {agg['lenient']['strict']}% "
                  f"(n={agg['n_tasks']}, reused {agg.get('n_reused',0)}) -> {out}", flush=True)


HDR = ("| model | method | SSR | full code func | body func | CSR | \u2800 | "
       "func_wrong (contract ok, func wrong) | over_rej (valid input rejected) | "
       "cvt_miss (invalid input accepted) | both_wrong | ssr_fail |")
SEP = "|" + "---|" * 12


def do_table(settings, models, scheme="lenient"):
    print(HDR); print(SEP)
    for M in models:
        for st in settings:
            f = os.path.join(STORE, M, f"{st}{SUFFIX}.json")
            if not os.path.exists(f):
                print(f"| {M} | {st} | not measured | | | | | | | | | |"); continue
            a = json.load(open(f))["aggregate"]; b = a[scheme]
            print(f"| {M} | {st} | {b['strict']:.2f}% | {a['full_code_func']:.2f}% | "
                  f"{a['body_func']:.2f}% | {b['csr']:.2f}% | | {b['fw']:.2f}% | {b['or']:.2f}% | "
                  f"{b['cm']:.2f}% | {b['bw']:.2f}% | {b['strict_fail']:.2f}% |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "table", "list"])
    ap.add_argument("settings", nargs="*", help="setting names; any name works, "
                    "it is the output/<setting>/ directory")
    ap.add_argument("--models", default="")
    ap.add_argument("--scheme", default="lenient", choices=["lenient", "strict_mode"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_reuse", action="store_true", help="re-execute even identical code")
    a = ap.parse_args()
    ms = a.models.split(",") if a.models else MODELS
    if a.cmd == "list":
        for st in SETTINGS:
            have = [M for M in MODELS if os.path.exists(os.path.join(STORE, M, f"{st}{SUFFIX}.json"))]
            print(f"  {st:10s} stored: {len(have)}/{len(MODELS)}  {have}")
    elif a.cmd == "score":
        do_score(a.settings or list(SETTINGS), ms, a.force, a.no_reuse)
    else:
        do_table(a.settings or list(SETTINGS), ms, a.scheme)
