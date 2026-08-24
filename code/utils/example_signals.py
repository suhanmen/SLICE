"""Recover executable example checks from the description — the ranking signal for selection.

The text is read from the task description itself, so no hidden test or contract-violating input
is involved. (Reading it from the masked functional view gives the same checks on every task --
masking never touches an example line -- but the description keeps the dependency chain shorter.) Three forms are parsed: bare `assert` lines, doctests
(`>>>` followed by the expected repr), and prose call notation (`f(a) = b`, turned into an assert).
A check must compile to be kept.

Each check is additionally annotated with whether the reference solution passes it. That flag is a
parse-quality diagnostic only — the selector never reads it, and a check the reference fails is
still used, because the model saw the same wrong example.

Writes artifacts/example_signals.json as {task id: {"checks": [{"form", "stmt"}], "n": int}}
"""
import json, re, os, sys, ast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "code")

FROZEN = set(json.load(open("dataset/contracteval/eval_tasks_340.json"))["ids"])
DATA = {str(r["id"]): r for r in json.load(open("dataset/contracteval/test.json"))}
_MASKED = None


def masked():
    """The base functional view, read on first use (artifacts/functional_view_base.json)."""
    global _MASKED
    if _MASKED is None:
        p = "artifacts/functional_view_base.json"
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} is missing — run scripts/build_artifacts.sh to rebuild it")
        _MASKED = json.load(open(p, encoding="utf-8"))
    return _MASKED

P_ASSERT = re.compile(r"^\s*(assert\s+.+)$", re.M)
P_DOCT = re.compile(r"^\s*>>>\s*(.+)$")
# prose call notation: entry(args) ==>|=>|➞|->|==|=|returns <value>
    # NOTE: longest separator first — a leading `==?` would swallow the `=` of `=>`
    #       and leave `> value`, which then fails to compile and is dropped.
def call_pat(entry):
    return re.compile(rf"({re.escape(entry)}\s*\([^\n]*?\))"
                      rf"\s*(?:==>|=>|\u279e|->|==|=|returns?)\s*([^\n]+)")


def compiles(stmt):
    try:
        ast.parse(stmt); return True
    except SyntaxError:
        return False


def parse_task(tid):
    r = DATA[tid]
    entry = r.get("entry_point") or ""
    txt = r.get("prompt_cs") or ""
    checks, seen = [], set()

    def add(form, stmt):
        stmt = stmt.strip().rstrip(".")
        if entry not in stmt or not compiles(stmt) or stmt in seen:
            return
        seen.add(stmt); checks.append({"form": form, "stmt": stmt})

    # 1) an assert line, used verbatim
    for m in P_ASSERT.finditer(txt):
        add("assert", m.group(1))
    # 2) doctest: the line after ">>> call" is the expected repr
    lines = txt.split("\n")
    for i, ln in enumerate(lines):
        m = P_DOCT.match(ln)
        if not m or entry not in m.group(1):
            continue
        call = m.group(1).strip()
        exp = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if exp and not exp.startswith(">>>") and not exp.startswith('"""'):
            add("doctest", f"assert {call} == {exp}")
    # 3) call notation inside prose
    for m in call_pat(entry).finditer(txt) if entry else []:
        call, exp = m.group(1), m.group(2).strip().rstrip(".,;")
        if exp.startswith(("=", ">")):        # mis-split "=="
            continue
        add("call", f"assert {call} == {exp}")
    return checks


def gold_pass(tid, checks):
    """Diagnostic only: does the reference solution pass these checks (i.e. did we mis-parse)?"""
    r = DATA[tid]
    sig, sol = r.get("signature") or "", r.get("canonical_solution") or ""
    if not sol.strip():
        return None
    prompt_code = r.get("prompt_base") or r.get("prompt_cs") or ""
    # ContractEval convention: the runnable reference is prompt stub + canonical solution
    full = prompt_code + "\n" + sol
    import io, contextlib, multiprocessing

    def run(q):
        env = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(full, env)
                flags = []
                for c in checks:
                    try:
                        exec(c["stmt"], env); flags.append(True)
                    except Exception:
                        flags.append(False)
            q.put(flags)
        except Exception:
            q.put(None)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run, args=(q,))
    p.start(); p.join(10)
    if p.is_alive():
        p.terminate(); return None
    try:
        return q.get_nowait()
    except Exception:
        return None


def main():
    out, cover = {}, {"with_checks": 0, "forms": {}, "total_checks": 0}
    gold_ok = gold_bad = gold_na = 0
    for tid in sorted(FROZEN):
        checks = parse_task(tid)
        out[tid] = {"checks": checks, "n": len(checks)}
        if checks:
            cover["with_checks"] += 1
            cover["total_checks"] += len(checks)
            for c in checks:
                cover["forms"][c["form"]] = cover["forms"].get(c["form"], 0) + 1
        flags = gold_pass(tid, checks) if checks else None
        if flags is None:
            gold_na += len(checks or [])
        else:
            for c, f in zip(checks, flags):
                c["gold"] = f          # diagnostic annotation; the selector ignores it
            gold_ok += sum(flags); gold_bad += len(flags) - sum(flags)
            out[tid]["gold_pass"] = sum(flags)

    os.makedirs("artifacts", exist_ok=True)
    json.dump(out, open("artifacts/example_signals.json", "w"), indent=1)
    n = len(FROZEN)
    print(f"coverage: {cover['with_checks']}/{n} tasks ({cover['with_checks']/n*100:.1f}%), "
          f"{cover['total_checks']} checks, forms {cover['forms']}")
    tot = gold_ok + gold_bad
    if tot:
        print(f"reference passes {gold_ok}/{tot} ({gold_ok/tot*100:.1f}%) — failures suggest a mis-parse")
    print(f"could not be checked (execution failed or timed out): {gold_na}")
    # a few examples of the failures, to see which form is at fault
    bad = [(t, c["stmt"][:70]) for t, v in out.items() if v.get("gold_pass") is not None
           and v["gold_pass"] < v["n"] for c in v["checks"]][:6]
    for t, s in bad:
        print("  reference-fails:", t, s)


if __name__ == "__main__":
    main()
