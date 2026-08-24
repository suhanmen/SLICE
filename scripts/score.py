#!/usr/bin/env python
"""Generic contract-eval scorer — one tool that replaces the ~16 one-off score_*.py.

Prints strict / func / CSR / csr_type / csr_value, then over_rejection ALWAYS split 2-way
(the split is computed in the same run, not a separate script):
  over_rej   = valid input hit an AssertionError (all of these are Contract X = over-guarded)
  ⊂ConX,FunO = assert wrongly rejects but body-only PASSES the input  -> true over-rejection
  ⊂ConX,FunX = assert wrongly rejects AND body-only also fails         -> really a func failure
`over_rej = ConX,FunO + ConX,FunX` (denominator = tasks with valid checks, matches the eval).

Run from the repository root, so `code/` and `dataset/` resolve:
    python scripts/score.py output/inline_full/contracteval/cs/<model>/phase2_INLINE_*.json
    python scripts/score.py "baseline=…/Qwen3.5-9B_output.json" "AUTO=…/gauto05.json" \
                            --subset-common --baseline …/Qwen3.5-9B_output.json
    python scripts/score.py <files> --no-split      # skip the 2-way (faster; single over_rej)
    python scripts/score.py <files> --decompose     # + contract-only / body-only rows
Each positional arg is a path, a glob, or `label=path`.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "code")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import persist                                          # noqa: E402
from utils.contract_eval import contract_evaluate, extract_python_code  # noqa: E402
from utils.dataset import load_contract_dataset      # noqa: E402
try:
    from utils.contract_eval import contract_only, strip_checks, _score_sample, _load_excluded_checks
    from utils.build_expected_outputs import load_or_build_expected
except Exception:                                     # pragma: no cover
    contract_only = strip_checks = _score_sample = _load_excluded_checks = load_or_build_expected = None

# aggregate metrics (from contract_evaluate); over_rej is shown from the split instead
COLS = [("strict", "strict_pass@1"), ("func", "func_total@1"), ("CSR", "csr@1"),
        ("csrT", "csr_type@1"), ("csrV", "csr_value@1")]
# (label, width). over_rej is task-level (the two subset columns sum to it);
# over_rej_TC is call-level, over all valid test cases.
SPLIT_COLS = [("over_rej", 10), ("⊂ Contract X, Func O", 22), ("⊂ Contract X, Func X", 22),
              ("over_rej_TC", 12)]

_CACHE = {}


def _split_ctx(dataset, root):
    if "cache" not in _CACHE:
        _CACHE["cache"] = load_or_build_expected(dataset, root, source="both")
        _CACHE["exc"] = _load_excluded_checks(dataset, root)
    return _CACHE["cache"], _CACHE["exc"]


def over_rej_split(recs, ids, dataset, root):
    """Returns (over_rej, ConX_FunO, ConX_FunX, over_rej_TC):
      first three  = TASK-level rates over tasks-with-checks (matches the eval's over_rej --
                     a problem counts if ANY of its valid test cases is over-rejected).
      over_rej_TC  = TEST-CASE-level rate: valid test cases the contract rejected,
                     over all valid test cases regardless of whether the body passes them.
    """
    cache, exc = _split_ctx(dataset, root)

    def checks_of(tid):
        ch = [[c["call"], c["expected_repr"], c["source"]] for c in cache.get(tid, {}).get("checks", [])]
        d = exc.get(tid)
        return [c for i, c in enumerate(ch) if not (d and i in d)] if d else ch

    def code_of(r):
        c = r.get("code") or ""
        return c if "def " in c else extract_python_code(c)

    denom = orej = con = ffl = 0
    tc_total = tc_orej = 0            # call level: denominator = every valid test case
    detail = {}                       # per-task verdicts, so a number stays traceable
    for r in recs:
        tid = str(r.get("id"))
        if ids is not None and tid not in ids:
            continue
        ch = checks_of(tid)
        if not ch:
            continue
        denom += 1
        full = _score_sample(code_of(r), ch, [], 15.0)
        oflags = list(full.get("orej_base", [])) + list(full.get("orej_plus", []))
        tc_total += len(oflags)
        tc_orej += sum(1 for x in oflags if x)
        n_orej = sum(1 for x in oflags if x)
        if not any(oflags):
            detail[tid] = {"over_rej": False, "n_cases": len(oflags), "n_orej": 0, "side": None}
            continue
        orej += 1
        body = _score_sample(strip_checks(code_of(r)), ch, [], 15.0)
        bpass = list(body.get("func_base", [])) + list(body.get("func_plus", []))
        is_con = any(oflags[i] and i < len(bpass) and bpass[i] for i in range(len(oflags)))
        con += int(is_con)
        ffl += int(not is_con)
        detail[tid] = {"over_rej": True, "n_cases": len(oflags), "n_orej": n_orej,
                       "side": "ConX_FunO" if is_con else "ConX_FunX"}
    d = max(1, denom)
    return orej / d, con / d, ffl / d, tc_orej / max(1, tc_total), detail


def expand(arg):
    if "=" in arg and "/" not in arg.split("=", 1)[0]:
        label, spec = arg.split("=", 1)
    else:
        label, spec = "", arg
    paths = sorted(glob.glob(spec)) or [spec]
    return [(label or Path(p).stem.replace("phase2_INLINE_", "").replace("_output", ""), p) for p in paths]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Generic contract-eval scorer (replaces score_*.py)")
    ap.add_argument("inputs", nargs="+", help="output json path(s), glob(s), or label=path")
    ap.add_argument("--dataset", default="contracteval")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--subset-common", action="store_true",
                    help="score all inputs on the intersection of their task ids")
    ap.add_argument("--no-split", action="store_true",
                    help="skip the 2-way over_rej split (single over_rejection_total, faster)")
    ap.add_argument("--decompose", action="store_true",
                    help="also print contract-only / body-only rows")
    ap.add_argument("--baseline", default="", help="path whose strict is subtracted -> vs column")
    # The result is always written to JSON (scripts/persist.py); this only changes where.
    ap.add_argument("--save", default="", help="result JSON path (default evaluation/scoring/)")
    args = ap.parse_args()
    do_split = not args.no_split and _score_sample is not None

    tasks = load_contract_dataset(args.tasks or f"dataset/{args.dataset}/test.json")
    loaded = [(lab, p, load(p)) for a in args.inputs for lab, p in expand(a)]

    ids = None
    if args.subset_common:
        idsets = [set(str(r.get("id")) for r in recs) for _, _, recs in loaded if recs]
        ids = set.intersection(*idsets) if idsets else set()
        print(f"# subset-common: {len(ids)} shared task ids")

    # `contract_evaluate` also returns per-task rows; keep the last call's rows so every
    # cell of the table can be traced back to the tasks behind it.
    last_rows = {}

    def agg(recs, transform=None, keep=None):
        sub = recs if ids is None else [r for r in recs if str(r.get("id")) in ids]
        if transform:
            sub = [{"id": r.get("id"), "code": transform(r.get("code", "") or "")} for r in sub]
        rep = contract_evaluate(sub, tasks, dataset=args.dataset,
                                project_root=args.project_root, ks=(1,))
        if keep:
            last_rows[keep] = rep.get("per_task") or []
        return rep["aggregate"], len(sub)

    base_strict = None
    if args.baseline and (brecs := load(args.baseline)):
        base_strict = agg(brecs)[0]["strict_pass@1"]

    orcols = SPLIT_COLS if do_split else [("over_rej", 10)]
    hdr = f"{'config':<26}" + "".join(f"{c:>9}" for c, _ in COLS) + "".join(f"{c:>{w}}" for c, w in orcols) + f"{'n':>6}"
    if base_strict is not None:
        hdr += f"{'vs_base':>9}"
    print(hdr); print("-" * len(hdr))

    def fmt(a, n, label, recs=None, indent=""):
        cells = "".join(f"{a.get(k, float('nan')):>9.3f}" for _, k in COLS)
        if do_split and recs is not None and not indent:
            orej, cx, fx, tc, det = over_rej_split(recs, ids, args.dataset, args.project_root)
            last_rows["split"] = det
            cells += f"{orej:>10.3f}{cx:>22.3f}{fx:>22.3f}{tc:>12.3f}"
        else:
            cells += f"{a.get('over_rejection_total@1', float('nan')):>10.3f}"
            if do_split:
                cells += " " * 56   # keep alignment under ⊂/⊂/TC columns for decompose rows
        line = f"{indent+label:<26}{cells}{n:>6}"
        if base_strict is not None and not indent:
            line += f"{a['strict_pass@1']-base_strict:>+9.3f}"
        print(line)

    dump = {}
    for lab, p, recs in loaded:
        if not recs:
            print(f"{lab:<26}(missing: {p})")
            dump[lab] = {"path": p, "missing": True}
            continue
        last_rows.pop("split", None)
        a, n = agg(recs, keep="main")
        fmt(a, n, lab, recs=recs)
        # Per task: join the contract_evaluate row with the over_rej split verdict by id.
        split = last_rows.get("split") or {}
        per_task = {}
        for row in last_rows.get("main", []):
            tid = str(row.get("id"))
            per_task[tid] = dict({k: v for k, v in row.items() if k != "id"},
                                 **(split.get(tid) or {}))
        dump[lab] = {"path": p, "n": n, "metrics": a, "per_task": per_task}
        if args.decompose and contract_only and strip_checks:
            ca, cn = agg(recs, contract_only, keep="contract_only")
            crows = {str(x.get("id")): x for x in last_rows.get("contract_only", [])}
            ba, bn = agg(recs, strip_checks, keep="body_only")
            brows = {str(x.get("id")): x for x in last_rows.get("body_only", [])}
            fmt(ca, cn, "· contract-only", indent="  ")
            fmt(ba, bn, "· body-only", indent="  ")
            dump[lab]["contract_only"] = {"n": cn, "metrics": ca, "per_task": crows}
            dump[lab]["body_only"] = {"n": bn, "metrics": ba, "per_task": brows}
    if do_split:
        print("\nover_rej (task level) = subset(Contract X, Func O) + subset(Contract X, Func X)")
        print("  Func O = body is correct, a true over-rejection | Func X = the body is wrong too")
        print("over_rej_TC (call level) = valid test cases rejected / all valid test cases")
    print("DONE")
    persist.save("score", {"dataset": args.dataset, "configs": dump},
                 override=args.save or None)


if __name__ == "__main__":
    main()
