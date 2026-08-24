"""Adapter: ContractEval.jsonl -> the task schema this repository uses.

Reads the ContractEval corpus (`ContractEval.jsonl`, obtain it from the original benchmark)
and writes dataset/<out>/test.json in the schema load_contract_dataset expects:
    {id, description, signature, valid_tests, cvts}

Mapping:
  id          <- task_id
  signature   <- the `def <entry_point>(...)` line from `prompt`
  description <- docstring text + the contract preconditions (contract_individual_NL)
  valid_tests <- base_input (+ optional plus_input): calls that must NOT raise
  cvts        <- contract_violating_test inputs: calls that SHOULD raise AssertionError

Caps are logged, never silently truncated. Run:
    python code/utils/build_contracteval_dataset.py --src <ContractEval.jsonl> \
        --out contracteval --max_valid 8 --max_cvt 8
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "dataset" / "ContractEval.jsonl"


def _extract_signature(rec: dict, entry_point: str) -> str:
    """Real `def {entry}(...)` line. HumanEval has it in `prompt`; MBPP doesn't (just a
    docstring), so fall back to `canonical_solution`, which carries the real def line."""
    for src in (rec.get("prompt", ""), rec.get("canonical_solution", ""),
                rec.get("canonical_solution_with_contract", "")):
        for line in (src or "").splitlines():
            if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", line):
                return line.strip().rstrip()
    return f"def {entry_point}(*args, **kwargs):"


def _extract_docstring(prompt: str) -> str:
    m = re.search(r'"""(.*?)"""', prompt, flags=re.DOTALL) or re.search(r"'''(.*?)'''", prompt, flags=re.DOTALL)
    if not m:
        return ""
    # collapse whitespace, drop doctest example lines
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().startswith(">>>")]
    return " ".join(lines)


def _base_description(rec: dict) -> str:
    """Base problem description only (NO contract info) — keeps base/CS/EAS separable."""
    return _extract_docstring(rec.get("prompt", ""))


def _contract_nl(rec: dict) -> str:
    """The contract preconditions in NL (used by the CS and EAS conditions)."""
    nl = rec.get("contract_individual_NL") or {}
    return " ".join(nl[k] for k in sorted(nl))


# The upstream ContractEval corpus ships ONE task with an empty `prompt_with_contract_aware`
# (HumanEval/20 — every other field is fine, and 363/364 tasks are complete). Rather than let it
# silently fall back to prompt_base (which would evaluate that task at the BASE level while it is
# labelled cs/eas), we restore the CS prompt by writing the contract into the docstring, in the
# same style upstream uses elsewhere — cf. HumanEval/21, whose contract is identical
# ("integers or floats", ">= 2 elements") and whose CS docstring reads:
#   "Given a list of numbers containing at least two elements where all elements are either
#    integers or floats, apply a linear transform ..."
# Authored restoration, not upstream data.
_CS_RESTORE = {
    "HumanEval/20": (
        "From a supplied list of numbers containing at least two elements where all elements "
        "are either integers or floats, select and return two that are the closest to each "
        "other and return them in order (smaller number, larger number)."
    ),
}


def _restore_prompt_cs(rec: dict, prompt_cs: str) -> str:
    """Fill in a missing prompt_with_contract_aware by rewriting the base docstring's first
    sentence with the contract folded in (upstream's own CS style). No-op when CS is present."""
    if prompt_cs.strip():
        return prompt_cs
    tid = str(rec.get("task_id"))
    new_first = _CS_RESTORE.get(tid)
    base = rec.get("prompt", "") or ""
    if not new_first or not base:
        return prompt_cs
    old_first = _base_description(rec).split(">>>")[0].strip()
    # the docstring text is wrapped across lines in `base`; rebuild it with the CS sentence
    m = re.search(r'("""\s*)(.*?)(\n\s*>>>|\s*""")', base, re.S)
    if not m:
        return prompt_cs
    return base[:m.start(2)] + new_first + base[m.end(2):]


def _valid_calls(entry_point: str, inputs: list) -> list:
    calls = []
    for args in inputs or []:
        if not isinstance(args, list):
            args = [args]
        calls.append(f"{entry_point}({', '.join(repr(a) for a in args)})")
    return calls


_EXEC_PREAMBLE = "from typing import *\nimport math, re, collections, itertools\n"


def _valid_checks(rec: dict, entry_point: str, inputs: list) -> list:
    """Run the canonical solution on each valid input to get the expected output, and
    return executable equality checks `entry(args) == <expected>` for functional pass@k.
    Tasks/inputs whose canonical fails are skipped (no check emitted)."""
    full = (rec.get("prompt", "") or "") + (rec.get("canonical_solution", "") or "")
    ns = {}
    try:
        exec(_EXEC_PREAMBLE + full, ns)
    except Exception:
        return []
    f = ns.get(entry_point)
    if not callable(f):
        return []
    checks = []
    for args in inputs or []:
        a = args if isinstance(args, list) else [args]
        try:
            out = f(*[__import__("copy").deepcopy(x) for x in a])
        except Exception:
            continue
        argstr = ", ".join(repr(x) for x in a)
        try:
            if eval(repr(out)) == out:           # only keep round-trippable expected values
                checks.append(f"{entry_point}({argstr}) == {out!r}")
        except Exception:
            continue
    return checks


def _param_names(signature: str) -> list:
    """Parse ordered parameter names from a `def name(a, b: int, c=5):` line."""
    m = re.search(r"\((.*)\)", signature, flags=re.DOTALL)
    if not m:
        return []
    names = []
    for part in m.group(1).split(","):
        name = part.split(":")[0].split("=")[0].strip().lstrip("*")
        if name and name not in ("self", "/"):
            names.append(name)
    return names


def _cvt_calls(entry_point: str, signature: str, cvts: list):
    """Return (calls, keys) — POSITIONAL calls so the model's parameter names don't have
    to match. The contract_violating_test dict is ordered by the signature's params."""
    params = _param_names(signature)
    calls, keys = [], []
    for item in cvts or []:
        inp = item.get("input", {}) if isinstance(item, dict) else {}
        if isinstance(inp, dict):
            ordered = [inp[p] for p in params if p in inp]
            ordered += [v for k, v in inp.items() if k not in params]  # defensive
            vals = ordered
        else:
            vals = inp if isinstance(inp, list) else [inp]
        calls.append(f"{entry_point}({', '.join(repr(v) for v in vals)})")
        keys.append(item.get("contract_in_key", "") if isinstance(item, dict) else "")
    return calls, keys


def _format_cvt_block(calls: list, keys: list) -> str:
    """Format CVTs exactly like ContractEval (TG_CG_main.py:482):
    `{key}:\n>>> {call}\n "AssertionError: invalid input"`, one per line."""
    lines = []
    for call, key in zip(calls, keys):
        lines.append(f'{key or "assert"}:\n>>> {call}\n "AssertionError: invalid input"')
    return "\n".join(lines)


def _build_prompt_eas(prompt_cs: str, eas_calls: list, eas_keys: list) -> str:
    """EAS prompt = the CS prompt + the shown-in-prompt CVTs (eas_prompt_cvts), matching
    ContractEval's contract-test-case prompt (1 CVT per clause)."""
    block = _format_cvt_block(eas_calls, eas_keys)
    if not block:
        return prompt_cs
    return f"{prompt_cs}\nContract Test Cases (must be rejected):\n{block}"


def _split_cvts(calls: list, keys: list):
    """Split CVTs into (eas_prompt, private) so the EAS prompt and the CSR eval are DISJOINT
    (no leakage). Grouped by contract clause (contract_in_key): if a clause has >=2 CVTs, ONE
    goes to the prompt (as an example) and the rest to private; a clause with a single CVT goes
    entirely to private (never shown in the prompt). => private is non-empty whenever any CVT
    exists, and every prompt CVT has a disjoint held-out sibling for the same clause."""
    from collections import OrderedDict
    groups = OrderedDict()
    for c, k in zip(calls, keys):
        groups.setdefault(k, []).append(c)
    p_calls, p_keys, v_calls, v_keys = [], [], [], []
    for k, g in groups.items():
        if len(g) >= 2:
            p_calls.append(g[0]); p_keys.append(k)
            for c in g[1:]:
                v_calls.append(c); v_keys.append(k)
        else:
            v_calls.append(g[0]); v_keys.append(k)
    return p_calls, p_keys, v_calls, v_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default="contracteval", help="dataset directory name under dataset/")
    p.add_argument("--valid_source", choices=["base", "plus", "both"], default="both",
                   help="which valid inputs build valid_tests/valid_checks: base=base_input "
                        "(public), plus=plus_input (EvalPlus large set), both=base+plus.")
    p.add_argument("--max_valid", type=int, default=0,
                   help="cap on valid_tests/valid_checks per task (0 = ALL, no sampling). Raw "
                        "base_input/plus_input are stored FULLY regardless.")
    p.add_argument("--max_cvt", type=int, default=0,
                   help="cap on cvts (0 = ALL, no truncation). Applies to the full cvts list; "
                        "the eas_prompt/private split is derived from the kept cvts.")
    p.add_argument("--limit", type=int, default=0, help="cap #tasks (0=all)")
    args = p.parse_args()

    src = Path(args.src)
    tasks = []
    n_base = n_plus = n_cvt = n_prompt_cvt = n_priv_cvt = 0
    empty_priv = empty_prompt = 0
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ep = rec["entry_point"]
            sig = _extract_signature(rec, ep)
            base = rec.get("base_input") or []
            plus = rec.get("plus_input") or []
            # valid inputs source is configurable: base (public), plus (EvalPlus large set), or
            # both. base is listed first so it isn't dropped if a cap is applied.
            # ALL cvts (uncapped unless --max_cvt>0), then split into eas-prompt vs private
            cvt, cvt_keys = _cvt_calls(ep, sig, rec.get("contract_violating_test"))
            if args.max_cvt:
                cvt, cvt_keys = cvt[: args.max_cvt], cvt_keys[: args.max_cvt]
            p_cvt, p_keys, v_cvt, v_keys = _split_cvts(cvt, cvt_keys)
            n_base += len(base); n_plus += len(plus)
            n_cvt += len(cvt); n_prompt_cvt += len(p_cvt); n_priv_cvt += len(v_cvt)
            empty_priv += (len(v_cvt) == 0 and len(cvt) > 0)
            empty_prompt += (len(p_cvt) == 0 and len(cvt) > 0)
            p_cs = _restore_prompt_cs(
                rec, (rec.get("prompt_with_contract_aware", "") or "").strip())
            if not (rec.get("prompt_with_contract_aware") or "").strip():
                print(f"[restore] {rec['task_id']}: prompt_cs was empty upstream -> "
                      f"{'RESTORED' if p_cs.strip() else 'STILL EMPTY'}")
            tasks.append({
                "id": rec["task_id"],
                "entry_point": ep,
                "prompt_base": (rec.get("prompt", "") or "").strip(),
                "prompt_cs": p_cs,
                # prebuilt EAS prompt = CS + eas_prompt_cvts (1 CVT/clause, ContractEval-style)
                "prompt_eas": _build_prompt_eas(p_cs, p_cvt, p_keys),
                "description": _base_description(rec),
                "contract_nl": _contract_nl(rec),
                "signature": sig,
                # canonical reference solution (right after signature). Expected outputs are NOT
                # stored (they blow up the file for large plus sets); eval computes them on demand
                # from prompt_base+canonical_solution and caches to inspect_output_pair.json.
                "canonical_solution": (rec.get("canonical_solution", "") or ""),
                # RAW public/private valid inputs from the source, in FULL (no truncation) —
                # so functional eval can be as rigorous as EvalPlus intends.
                "base_input": base,
                "plus_input": plus,
                # FULL contract-violating tests + keys
                "cvts": cvt,
                "cvt_keys": cvt_keys,
                # EAS-prompt subset (shown to the model) — DISJOINT from private
                "eas_prompt_cvts": p_cvt,
                "eas_prompt_cvt_keys": p_keys,
                # held-out CVTs for CSR eval (no leakage): score CSR on THESE, not on cvts shown
                "private_cvts": v_cvt,
                "private_cvt_keys": v_keys,
            })
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = PROJECT_ROOT / "dataset" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    n = len(tasks)
    print(f"[build] wrote {n} tasks -> {out_path}")
    print(f"[build] raw base_input total={n_base} (avg {n_base/n:.1f}), "
          f"plus_input total={n_plus} (avg {n_plus/n:.1f})")
    print(f"[build] expected outputs NOT stored (computed at eval from canonical_solution -> "
          f"inspect_output_pair.json cache). valid_source={args.valid_source} governs eval inputs.")
    print(f"[build] cvts total={n_cvt} (avg {n_cvt/n:.1f}) -> "
          f"eas_prompt={n_prompt_cvt} (avg {n_prompt_cvt/n:.1f}), "
          f"private(eval)={n_priv_cvt} (avg {n_priv_cvt/n:.1f})")
    print(f"[build] tasks with empty private_cvts={empty_priv} (CSR uneval-able), "
          f"empty eas_prompt_cvts={empty_prompt} (EAS==CS for those, all clauses single-CVT)")


if __name__ == "__main__":
    main()
