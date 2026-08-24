"""Expected-output cache builder for the contract dataset.

test.json stores only the valid INPUTS (base_input / plus_input) + canonical_solution — NOT
the expected outputs (materializing them for the large EvalPlus `plus` set blows the file up
to >1GB). Instead the functional-correctness reference is computed ONCE by running the
canonical solution on every valid input and cached to `inspect_output_pair.json` next to
test.json. Eval loads that cache (and regenerates it if missing) instead of re-running the
canonical every time.

Cache schema (inspect_output_pair.json):
    { "<task_id>": {
        "entry_point": str,
        "checks": [ {"call": "f(args)", "expected_repr": "<repr(out)>", "source": "base|plus"} ]
      }, ... }
`call` + `expected_repr` compose the executable check `assert (call) == expected_repr`.
Inputs where the canonical raises or whose output isn't round-trippable (repr!=value) are
skipped (no check emitted), exactly like the old inline _valid_checks.

Run:  python code/utils/build_expected_outputs.py --dataset contracteval [--source base|plus|both]
"""

import argparse
import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXEC_PREAMBLE = ("from typing import List, Tuple, Dict, Optional, Any, Set, Union\n"
                  "import math, re, collections, itertools\n")


def _canonical_fn(task: dict):
    """exec prompt_base + canonical_solution -> the reference callable (or None)."""
    src = (task.get("prompt_base", "") or "") + "\n" + (task.get("canonical_solution", "") or "")
    ns = {}
    try:
        exec(_EXEC_PREAMBLE + src, ns)
    except Exception:
        return None
    f = ns.get(task.get("entry_point"))
    return f if callable(f) else None


def _checks_for(task: dict, f, inputs: list, source: str) -> list:
    ep = task.get("entry_point")
    out = []
    for args in inputs or []:
        a = args if isinstance(args, list) else [args]
        try:
            val = f(*[copy.deepcopy(x) for x in a])
        except Exception:
            continue                       # canonical raises on this input -> skip
        try:
            if eval(repr(val)) != val:     # keep only round-trippable expected values
                continue
        except Exception:
            continue
        out.append({"call": f"{ep}({', '.join(repr(x) for x in a)})",
                    "expected_repr": repr(val), "source": source})
    return out


def build_expected_outputs(dataset: str, project_root: str = ".", source: str = "both") -> Path:
    root = Path(project_root)
    ds_dir = root / "dataset" / dataset
    tasks = json.load(open(ds_dir / "test.json", encoding="utf-8"))
    cache, n_checks, n_no_fn = {}, 0, 0
    for t in tasks:
        f = _canonical_fn(t)
        if f is None:
            n_no_fn += 1
            cache[t["id"]] = {"entry_point": t.get("entry_point"), "checks": []}
            continue
        checks = []
        if source in ("base", "both"):
            checks += _checks_for(t, f, t.get("base_input"), "base")
        if source in ("plus", "both"):
            checks += _checks_for(t, f, t.get("plus_input"), "plus")
        n_checks += len(checks)
        cache[t["id"]] = {"entry_point": t.get("entry_point"), "checks": checks}
    out_path = ds_dir / "inspect_output_pair.json"
    json.dump(cache, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[expected] {len(tasks)} tasks, {n_checks} checks (source={source}), "
          f"{n_no_fn} tasks canonical failed to load -> {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


_CACHE_MEM = {}  # {resolved_path: dict} — the 1.6GB cache is loaded once per process


def load_or_build_expected(dataset: str, project_root: str = ".", source: str = "both") -> dict:
    """Return the expected-output cache dict, building it if the file is absent. Memoized in
    process so repeated eval calls (e.g. eval_compare scoring 6 conditions) load it once."""
    p = (Path(project_root) / "dataset" / dataset / "inspect_output_pair.json").resolve()
    if not p.exists():
        build_expected_outputs(dataset, project_root, source)
    key = str(p)
    if key not in _CACHE_MEM:
        _CACHE_MEM[key] = json.load(open(p, encoding="utf-8"))
    return _CACHE_MEM[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="contracteval")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--source", choices=["base", "plus", "both"], default="both",
                    help="which valid inputs to compute expected outputs for")
    a = ap.parse_args()
    build_expected_outputs(a.dataset, a.project_root, a.source)


if __name__ == "__main__":
    main()
