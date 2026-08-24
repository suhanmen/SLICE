"""Contract-satisfaction evaluation.

Computes the metrics the contract setting requires:

  - CSR (Contract Satisfaction Rate): fraction of CVTs (invalid inputs) that are
    rejected by an *intended assertion* (AssertionError), NOT an incidental crash /
    silent accept / valid over-reject.
  - pass@1 retention: fraction of valid-input tests that still run without raising
    (the validator must not break functionality / over-reject).
  - cost-normalized: CSR per guided (validator) token.

Execution is sandboxed in a subprocess with a timeout. Outcomes are classified:
  ok        -> ran, no exception            (wanted for valid_tests)
  assertion -> raised AssertionError        (wanted for cvts = intended rejection)
  error     -> raised some other exception  (incidental crash; NOT contract success)
  timeout   -> exceeded the time budget

Usage:
    python code/utils/contract_eval.py --results <output.json> --dataset <name>
    # or programmatically: contract_evaluate(results, tasks)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Common imports so type-hinted signatures (List[str], etc.) and exec'd code don't NameError.
_PREAMBLE = ("from typing import List, Tuple, Dict, Optional, Any, Set, Union\n"
             "import math, re, collections, itertools\n")


def run_snippet(code: str, snippet: str, timeout: float = 5.0) -> str:
    """Exec `code` then `snippet` in a fresh subprocess; return the outcome label."""
    script = _PREAMBLE + code + "\n" + snippet + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if proc.returncode == 0:
        return "ok"
    return "assertion" if "AssertionError" in proc.stderr else "error"


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c, k)/C(n, k)."""
    from math import comb
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


_FLOAT_LIT = re.compile(r"(?<![\w.])\d+\.\d+")


def _cvt_is_type_violation(cvt: str) -> bool:
    """Heuristic difficulty split: a float literal where an int is expected is a TYPE
    violation, catchable by a single `isinstance` assert (~70% of contracteval CVTs = easy).
    Everything else is treated as a VALUE/range/relation violation (harder — needs a real
    predicate). Rough; a finer split would compare CVT arg types vs valid_test arg types."""
    return bool(_FLOAT_LIT.search(cvt))


def _sample_outcomes(code: str, task: dict, timeout: float = 5.0):
    """For one generated function, return
    (func_ok, retention_ok, contract_ok, csr_type_ok, csr_value_ok) booleans (None when the
    task has no tests of that kind):
      func_ok      : ALL valid_checks pass (output == expected)  -> functional correctness
      retention_ok : ALL valid_tests run without raising         -> not over-rejecting
      contract_ok  : ALL cvts raise AssertionError               -> contract satisfaction (CSR)
      csr_type_ok  : ALL *type-violation* CVTs rejected          -> easy CVTs (isinstance)
      csr_value_ok : ALL *value/range* CVTs rejected             -> hard CVTs
    """
    checks = task.get("valid_checks", [])
    valids = task.get("valid_tests", [])
    cvts = task.get("cvts", [])
    func_ok = (all(run_snippet(code, f"assert ({c})", timeout) == "ok" for c in checks)
               if checks else None)
    retention_ok = (all(run_snippet(code, s, timeout) == "ok" for s in valids)
                    if valids else None)
    # execute each CVT once, then split by difficulty
    rej = {s: (run_snippet(code, s, timeout) == "assertion") for s in cvts}
    type_cvts = [s for s in cvts if _cvt_is_type_violation(s)]
    value_cvts = [s for s in cvts if not _cvt_is_type_violation(s)]
    contract_ok = (all(rej[s] for s in cvts) if cvts else None)
    csr_type_ok = (all(rej[s] for s in type_cvts) if type_cvts else None)
    csr_value_ok = (all(rej[s] for s in value_cvts) if value_cvts else None)
    return func_ok, retention_ok, contract_ok, csr_type_ok, csr_value_ok


def _samples_of(r: dict):
    return r.get("samples") or ([r["code"]] if r.get("code") is not None else [])


# ── Batched per-sample scoring ──────────────────────────────────────────────────────────
# One subprocess per (task, sample) runs the generated code against ALL of the task's valid
# checks (base+plus, from the inspect_output_pair cache) and ALL private_cvts, instead of one
# subprocess per test (which is infeasible for the ~300 plus inputs/task). It reports, per
# source (base/plus): func (output==expected) and valid_pass (ran without raising = NOT
# over-rejected); and per private CVT: the rejection outcome (assertion/ok/error).
_BATCH_RUNNER = r'''
import sys, json
job = json.load(sys.stdin)
ns = {}
_pre = "from typing import List, Tuple, Dict, Optional, Any, Set, Union\nimport math, re, collections, itertools\n"
try:
    exec(_pre + job["gen_code"], ns)
    loaded = True
except Exception:
    loaded = False
res = {"func_base": [], "func_plus": [], "orej_base": [], "orej_plus": [], "csr": [], "loaded": loaded}
if loaded:
    for call, exp, src in job["checks"]:
        orej = False
        try:
            got = eval(call, ns); ran = True
        except AssertionError:
            ran = False; got = None; orej = True   # valid input rejected by an assert = OVER-REJECTION
        except Exception:
            ran = False; got = None                # body crash on a valid input (NOT over-rejection)
        ok = False
        if ran:
            try:
                ok = (got == eval(exp, ns))
            except Exception:
                ok = False
        res["orej_" + src].append(bool(orej))
        res["func_" + src].append(bool(ok))
    for c in job["cvts"]:
        try:
            eval(c, ns); out = "ok"
        except AssertionError:
            out = "assertion"
        except Exception:
            out = "error"
        res["csr"].append(out)
print(json.dumps(res))
'''


def _score_sample(gen_code: str, checks: list, private_cvts: list, timeout: float) -> dict:
    """Run one generated program against all checks+cvts in a single subprocess.
    checks: [[call, expected_repr, source]]; private_cvts: [call_str]."""
    job = json.dumps({"gen_code": gen_code or "", "checks": checks, "cvts": private_cvts})
    try:
        proc = subprocess.run([sys.executable, "-c", _BATCH_RUNNER],
                              input=job, capture_output=True, text=True, timeout=timeout)
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        # timeout / crash / unparsable -> everything fails for this sample
        return {"func_base": [], "func_plus": [], "orej_base": [], "orej_plus": [],
                "csr": ["error"] * len(private_cvts), "loaded": False, "_dead": True}


def _all_true(xs):
    return (all(xs) if xs else None)


def _load_excluded_checks(dataset: str, project_root: str) -> dict:
    """Test cases the GOLD contract (canonical_solution_with_contract) itself rejects.

    Built by scripts/check_gold_contract.py: 3,066 / 111,011 checks (2.76%) across 29 tasks are
    contradictions in the ContractEval corpus — the attached contract excludes the problem's own
    valid inputs (21 tasks lose ALL their tests; 8 lose only some). No model, not even gold, can
    pass them, so they add a fixed floor to over_rejection/func. Excluded from func & over_rejection
    scoring; CVTs are unaffected (gold rejects 360/360 correctly), so CSR is untouched.
    {task_id: [check_index, ...]}; absent file -> no exclusion.
    """
    p = Path(project_root) / "dataset" / dataset / "excluded_checks.json"
    try:
        return {k: set(v) for k, v in json.loads(p.read_text()).items()}
    except Exception:
        return {}


def contract_evaluate(results: list, tasks: list, dataset: str = "contracteval",
                      project_root: str = ".", ks=(1,), timeout: float = 20.0,
                      desc: str = None) -> dict:
    """Score generations with the NEW protocol:
      - func_base@k / func_plus@k : functional correctness on base_input / plus_input SEPARATELY
        (output==expected), expected outputs from the inspect_output_pair cache.
      - valid_pass_base@1 / valid_pass_plus@1 : valid inputs that ran WITHOUT raising = NOT
        over-rejected (the metric formerly called 'retention').
      - csr@k / csr_type@k / csr_value@k : contract satisfaction on PRIVATE_CVTS (held-out, no
        leakage): all private CVTs rejected by an intended AssertionError.
    Batched: one subprocess per (task, sample). `dataset`/`project_root` locate the cache.
    """
    from utils.build_expected_outputs import load_or_build_expected
    cache = load_or_build_expected(dataset, project_root, source="both")
    excluded = _load_excluded_checks(dataset, project_root)
    by_id = {str(t.get("id")): t for t in tasks}
    ks = sorted(set(int(k) for k in ks))
    acc = {f"func_base@{k}": [] for k in ks}
    acc.update({f"func_plus@{k}": [] for k in ks})
    acc.update({f"func_total@{k}": [] for k in ks})   # base AND plus all pass (EvalPlus-style)
    acc.update({f"strict_pass@{k}": [] for k in ks})  # func_total AND all private CVTs rejected
    acc.update({f"csr@{k}": [] for k in ks})
    acc.update({f"csr_type@{k}": [] for k in ks})
    acc.update({f"csr_value@{k}": [] for k in ks})
    orej_b, orej_p, orej_t, tok_costs, rows = [], [], [], [], []

    # build per-task metadata + a flat list of (task_idx, sample_idx, code) scoring jobs
    metas, jobs = [], []
    for r in results:
        task = by_id.get(str(r.get("id")))
        if task is None:
            continue
        entry = cache.get(str(r.get("id")), {})
        checks = [[c["call"], c["expected_repr"], c["source"]] for c in entry.get("checks", [])]
        # drop the test cases the GOLD contract itself rejects (dataset contradictions) — they are
        # unpassable for any model, gold included, so they only add a floor to over_rejection.
        drop = excluded.get(str(r.get("id")))
        if drop:
            checks = [c for i, c in enumerate(checks) if i not in drop]
        priv = task.get("private_cvts") or []
        priv_type = [_cvt_is_type_violation(s) for s in priv]
        samples = _samples_of(r)
        m = {"task": task, "checks": checks, "priv": priv, "priv_type": priv_type,
             "n": len(samples), "tok": r.get("n_validator_tokens"),
             "has_base": any(c[2] == "base" for c in checks),
             "has_plus": any(c[2] == "plus" for c in checks),
             "has_cvt": bool(priv), "has_type": any(priv_type),
             "has_value": any(not x for x in priv_type)}
        ti = len(metas)
        metas.append(m)
        for si, code in enumerate(samples):
            jobs.append((ti, si, code))

    # score every (task, sample) concurrently — each is one subprocess, so threads (which just
    # wait on subprocess I/O) give real parallelism up to ~#cores.
    workers = max(1, min(16, (os.cpu_count() or 4) - 2))
    outcomes = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_score_sample, code, metas[ti]["checks"], metas[ti]["priv"], timeout):
                (ti, si) for (ti, si, code) in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=(desc or "eval"),
                        unit="samp", leave=False):
            outcomes[futs[fut]] = fut.result()

    def _csr_ok(x, mask=None):
        outs = x["csr"]
        idx = [i for i in range(len(outs)) if (mask is None or mask[i])]
        return all(outs[i] == "assertion" for i in idx) if idx else None

    _all_ok = lambda xs: (all(xs) if xs else True)   # vacuously ok when no tests of that kind
    for ti, m in enumerate(metas):
        per = [outcomes[(ti, si)] for si in range(m["n"])]
        n = m["n"]
        c_fb = sum(1 for x in per if _all_true(x["func_base"]))
        c_fp = sum(1 for x in per if _all_true(x["func_plus"]))
        # total = code LOADS and ALL base AND ALL plus functionality tests pass (EvalPlus-style).
        # `loaded` guard is essential: a parse-failed sample has empty check lists, and
        # _all_ok([]) is vacuously True — without the guard those would count as passing.
        c_ft = sum(1 for x in per
                   if x.get("loaded") and _all_ok(x["func_base"]) and _all_ok(x["func_plus"]))
        c_csr = sum(1 for x in per if _csr_ok(x))
        c_type = sum(1 for x in per if _csr_ok(x, m["priv_type"]))
        c_value = sum(1 for x in per if _csr_ok(x, [not t for t in m["priv_type"]]))
        # strict = the SAME code passes ALL functionality (base AND plus) AND rejects ALL
        # private CVTs — i.e. functionally correct AND contract-satisfying at once.
        c_strict = sum(1 for x in per if x.get("loaded") and _all_ok(x["func_base"])
                       and _all_ok(x["func_plus"]) and (_csr_ok(x) is True))
        for k in ks:
            if m["has_base"]:
                acc[f"func_base@{k}"].append(_pass_at_k(n, c_fb, k))
            if m["has_plus"]:
                acc[f"func_plus@{k}"].append(_pass_at_k(n, c_fp, k))
            if m["has_base"] or m["has_plus"]:
                acc[f"func_total@{k}"].append(_pass_at_k(n, c_ft, k))
            if (m["has_base"] or m["has_plus"]) and m["has_cvt"]:
                acc[f"strict_pass@{k}"].append(_pass_at_k(n, c_strict, k))
            if m["has_cvt"]:
                acc[f"csr@{k}"].append(_pass_at_k(n, c_csr, k))
            if m["has_type"]:
                acc[f"csr_type@{k}"].append(_pass_at_k(n, c_type, k))
            if m["has_value"]:
                acc[f"csr_value@{k}"].append(_pass_at_k(n, c_value, k))
        # over-rejection = the 1st sample rejects ANY valid input with an AssertionError
        if per and m["has_base"]:
            orej_b.append(1.0 if any(per[0]["orej_base"]) else 0.0)
        if per and m["has_plus"]:
            orej_p.append(1.0 if any(per[0]["orej_plus"]) else 0.0)
        if per and (m["has_base"] or m["has_plus"]):
            orej_t.append(1.0 if (any(per[0]["orej_base"]) or any(per[0]["orej_plus"])) else 0.0)
        if m["tok"]:
            tok_costs.append(m["tok"])
        # Per-task rows, so any number can be traced back to the task that produced it.
        # Every value below is already computed above, so this costs no extra execution.
        rows.append({"id": m["task"].get("id"), "n_samples": n, "c_func_base": c_fb,
                     "c_func_plus": c_fp, "c_csr": c_csr,
                     "c_func_total": c_ft, "c_strict": c_strict,
                     "c_csr_type": c_type, "c_csr_value": c_value,
                     "n_checks": len(m["checks"]), "n_cvt": len(m["priv"]),
                     "over_rej": bool(per and (any(per[0]["orej_base"])
                                               or any(per[0]["orej_plus"]))),
                     "tok": m["tok"]})

    mean = lambda xs: sum(xs) / len(xs) if xs else None
    agg = {"n_tasks": len(rows)}
    for k in ks:
        agg[f"func_base@{k}"] = mean(acc[f"func_base@{k}"])
        agg[f"func_plus@{k}"] = mean(acc[f"func_plus@{k}"])
        agg[f"func_total@{k}"] = mean(acc[f"func_total@{k}"])  # base AND plus all pass
        agg[f"strict_pass@{k}"] = mean(acc[f"strict_pass@{k}"])  # func_total AND all CVTs rejected
        agg[f"csr@{k}"] = mean(acc[f"csr@{k}"])
        agg[f"csr_type@{k}"] = mean(acc[f"csr_type@{k}"])     # easy (type violations)
        agg[f"csr_value@{k}"] = mean(acc[f"csr_value@{k}"])   # hard (value/range) — real signal
    # over_rejection = fraction of tasks where a FUNCTIONALITY TEST CASE (base_input/plus_input)
    # is rejected by an AssertionError (the validator wrongly rejecting it — NOT a body crash).
    # HIGHER IS WORSE. base/plus/total (base OR plus) reported.
    agg["over_rejection_base@1"] = mean(orej_b)
    agg["over_rejection_plus@1"] = mean(orej_p)
    agg["over_rejection_total@1"] = mean(orej_t)
    # func_fail = PURE functionality failure (wrong output or non-assert crash), i.e. func fails
    # for a reason OTHER than over-rejection. Since over_rejection ⟹ func fail, the three are a
    # partition:  func_pass + over_rejection + func_fail = 1.  HIGHER IS WORSE.
    def _ff(f, o):
        return max(0.0, 1.0 - f - o) if isinstance(f, float) and isinstance(o, float) else None
    agg["func_fail_base@1"] = _ff(agg.get("func_base@1"), agg.get("over_rejection_base@1"))
    agg["func_fail_plus@1"] = _ff(agg.get("func_plus@1"), agg.get("over_rejection_plus@1"))
    agg["func_fail_total@1"] = _ff(agg.get("func_total@1"), agg.get("over_rejection_total@1"))
    agg["n_type_tasks"] = len(acc[f"csr_type@{ks[0]}"])
    agg["n_value_tasks"] = len(acc[f"csr_value@{ks[0]}"])
    agg["mean_validator_tokens"] = mean(tok_costs)
    return {"aggregate": agg, "per_task": rows}


# --------------------------------------------------------------------------------------
# Decomposition: split a full solution into its CONTRACT (asserts) and BODY (functionality)
# parts, so a baseline solution can be scored the same way the method is. contract-only -> CSR
# (how well the contract was generated); body-only -> func_pass@1 (how well the body was
# generated, with the assertions' over-rejection removed).
# --------------------------------------------------------------------------------------
import ast as _ast


def _first_funcdef(tree):
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef):
            return node
    return None


class _RmChecks(_ast.NodeTransformer):
    """Delete assert/raise statements (leaves the functionality body)."""
    def visit_Assert(self, node):
        return None

    def visit_Raise(self, node):
        return None


def _strip_leading_guards(body):
    """Drop a function's ENTRY-REGION input-validation code (leading asserts + if/for/while checks),
    keeping a leading docstring and everything from the first PRODUCTIVE statement on. This
    preserves a FUNCTIONAL `raise`/`assert` used as algorithm logic (e.g. `raise ValueError('no
    repeat')` at the end of a function, or a broken assert that is not a contract) — those are
    not entry validation, so they stay. Contract validation lives at the function entry, so
    removing the leading validation statements targets exactly that."""
    out, stripping = [], True
    for i, s in enumerate(body):
        if (i == 0 and isinstance(s, _ast.Expr) and isinstance(getattr(s, "value", None), _ast.Constant)
                and isinstance(s.value.value, str)):
            out.append(s); continue                      # keep leading docstring
        if stripping and (isinstance(s, _ast.Assert) or _is_guard(s)):
            continue                                     # entry-region validation statement -> drop
        stripping = False                                # first productive stmt -> keep the rest
        out.append(s)
    return out or [_ast.Pass()]


def strip_checks(code: str) -> str:
    """BODY-ONLY: remove the ENTRY-REGION input-validation code (leading asserts + if/for/while
    checks) from every function; keep everything from the first productive statement on. Unlike a
    blanket assert/raise delete, this preserves a FUNCTIONAL `raise` used as body logic (so the
    body's real correctness is measured, not an artificially 'fixed' body). Falls back to a line
    filter if the code doesn't parse."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return "\n".join(ln for ln in code.splitlines()
                         if not (ln.strip().startswith(("assert ", "raise ")) or ln.strip() == "assert"))
    for fn in [n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)]:
        fn.body = _strip_leading_guards(fn.body)
    _ast.fix_missing_locations(tree)
    try:
        return _ast.unparse(tree)
    except Exception:
        return code


def _is_guard(stmt) -> bool:
    """An `if`/`for`/`while` whose only effect is to assert/raise (a validation guard),
    e.g. `if not isinstance(x, int): raise ...`."""
    if not isinstance(stmt, (_ast.If, _ast.For, _ast.While)):
        return False
    inner = list(getattr(stmt, "body", [])) + list(getattr(stmt, "orelse", []))
    return len(inner) > 0 and all(
        isinstance(s, (_ast.Assert, _ast.Raise, _ast.Pass)) for s in inner)


def _prune_to_checks(stmts):
    """Recursively keep assert/raise and the control-flow blocks (for/while/if/with/try) that
    contain them, PRESERVING the block headers (so a loop variable an inner assert references,
    e.g. `for num in x: assert isinstance(num, int)`, stays defined). Productive 'work'
    statements (assignments, returns, bare calls) are dropped."""
    out = []
    for s in stmts:
        if isinstance(s, (_ast.Assert, _ast.Raise)):
            out.append(s)
        elif isinstance(s, (_ast.For, _ast.While, _ast.If, _ast.With, _ast.Try)):
            if any(isinstance(n, (_ast.Assert, _ast.Raise)) for n in _ast.walk(s)):
                for attr in ("body", "orelse", "finalbody"):
                    b = getattr(s, attr, None)
                    if isinstance(b, list):
                        setattr(s, attr, _prune_to_checks(b) or [_ast.Pass()])
                out.append(s)
        # else: drop (the answer-producing body)
    return out


def contract_only(code: str) -> str:
    """CONTRACT-ONLY: keep the validation statements (assert/raise) IN THEIR ORIGINAL STRUCTURE
    — including the enclosing for/while/if blocks, so loop variables the asserts reference stay
    defined — and drop the answer-producing body, ending with `return None`. This runs the SAME
    asserts as the full code for precondition / element-wise checks (contract over_rejection/CSR
    then match full). Post-condition asserts that depend on a value the body ACCUMULATES still
    can't be isolated (their inputs are gone) — but body over_rejection is 0, so the full row IS
    the contract's over_rejection for those."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code
    fn = _first_funcdef(tree)
    if fn is None:
        return code
    fn.body = _prune_to_checks(fn.body) + [_ast.Return(value=_ast.Constant(value=None))]
    _ast.fix_missing_locations(tree)
    try:
        return _ast.unparse(tree)
    except Exception:
        return code


def decompose_evaluate(results: list, tasks: list, ks=(1,), timeout: float = 5.0,
                       desc: str = None) -> dict:
    """Score the CONTRACT part and the BODY part of each solution separately.
    Returns {"contract": <report on contract_only>, "body": <report on body_only>} where each
    is a normal contract_evaluate report. Read: contract.csr* = contract-generation quality;
    body.func_pass@1 = body-generation quality (validator over-rejection removed)."""
    contract_recs, body_recs = [], []
    for r in results:
        samples = _samples_of(r)
        contract_recs.append({"id": r.get("id"),
                              "samples": [contract_only(c) for c in samples]})
        body_recs.append({"id": r.get("id"),
                         "samples": [strip_checks(c) for c in samples]})
    d = f"{desc} " if desc else ""
    return {
        "contract": contract_evaluate(contract_recs, tasks, ks=ks, timeout=timeout,
                                      desc=f"{d}contract"),
        "body": contract_evaluate(body_recs, tasks, ks=ks, timeout=timeout, desc=f"{d}body"),
    }


def extract_python_code(md: str) -> str:
    """Extract code from a model response. Mirrors ContractEval's extract_python_code
    (```python ... ``` blocks); falls back to a bare ``` block, then to the raw text."""
    import re
    for pat in (r"```python\s*(.*?)```", r"```\s*(.*?)```"):
        blocks = re.findall(pat, md, re.DOTALL)
        if blocks:
            return "\n\n".join(b.strip() for b in blocks)
    return md.strip()


def _load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="path to a *_output.json")
    p.add_argument("--dataset", required=True, help="contract dataset name or json path")
    p.add_argument("--ks", default="1", help="comma-separated k values for pass@k")
    p.add_argument("--timeout", type=float, default=5.0)
    args = p.parse_args()

    results = _load_results(args.results)
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        ds_path = PROJECT_ROOT / "dataset" / args.dataset / "test.json"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.dataset import load_contract_dataset
    tasks = load_contract_dataset(ds_path)

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    report = contract_evaluate(results, tasks, ks=ks, timeout=args.timeout)
    print(json.dumps(report["aggregate"], indent=2))
