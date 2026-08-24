"""AST insertion of the contract assertions, and the deterministic template arm.

`insert_asserts` is what stage (iii) calls: it takes the assert lines the model produced, keeps the
ones that parse and reference an actual parameter, drops duplicates and inserts them after the
docstring.

They are also ordered by dependency, so that a length or comparison check never runs before the
type check that protects it (a wrong-typed input would raise TypeError, and a crash is not a
rejection). Measured on this benchmark the ordering changes no task-level score — the models
already emit the checks in a workable order — so it is a safeguard, not a contribution.

`attach` is the comparison arm: the same insertion, but the asserts are composed from the rule
parser's JSON specification (artifacts/condition_specs.json) instead of being generated.

    python code/utils/attach_assertions.py <selected.json> <out.json>
"""
import json, sys, os, ast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
def _load(name):
    """Read an artifacts/ file on first use, with a pointer to the script that rebuilds it."""
    import json as _json, os as _os
    p = _os.path.join("artifacts", name)
    if not _os.path.exists(p):
        raise FileNotFoundError(f"{p} is missing — run scripts/build_artifacts.sh to rebuild it")
    return _json.load(open(p, encoding="utf-8"))


_SPECS = None


def specs():
    global _SPECS
    if _SPECS is None:
        _SPECS = _load("condition_specs.json")
    return _SPECS

ORDER = {"type": 0, "membership": 1, "elements_type": 1, "nonempty": 2, "range": 2}


def type_tuple(types):
    flat = []
    for t in types:
        if t == "(int, float)":
            flat += ["int", "float"]
        else:
            flat.append(t)
    flat = list(dict.fromkeys(flat))
    return flat[0] if len(flat) == 1 else "(" + ", ".join(flat) + ")"


def spec_to_assert(s):
    k = s["kind"]
    if k == "type":
        return f"assert isinstance({s['arg']}, {type_tuple(s['types'])})"
    if k == "elements_type":
        return (f"assert all(isinstance(_e, {type_tuple(s['types'])}) "
                f"for _e in {s['arg']})")
    if k == "nonempty":
        return f"assert len({s['arg']}) > 0"
    if k == "range":
        return f"assert {s['subject']} {s['op']} {s['bound']}"
    if k == "membership":
        conds = []
        chars = [a for a in s["allowed"] if not a.startswith("CLASS_")]
        if chars:
            charset = "".join(chars).replace('"', '\\"')
            conds.append(f'_c in "{charset}"')
        if "CLASS_ALPHA" in s["allowed"]:
            conds.append("_c.isalpha()")
        if "CLASS_DIGIT" in s["allowed"]:
            conds.append("_c.isdigit()")
        if not conds:
            return None
        return f"assert all({' or '.join(conds)} for _c in {s['arg']})"
    return None


def insert_asserts(code, entry, assert_lines):
    """Insert assert lines at the function entry.
    Returns (new_code, #inserted); on a parse failure (original code, -1)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, -1
    fn = fn_of(tree, entry)
    if fn is None:
        return code, -1
    fn_args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.posonlyargs} \
              | {a.arg for a in fn.args.kwonlyargs}
    stmts, seen = [], set()
    for a in assert_lines:
        a = a.strip()
        if not a.startswith("assert") or a in seen:
            continue
        try:
            node = ast.parse(a).body[0]
        except SyntaxError:
            continue
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if not (names & fn_args):        # drop asserts that reference no parameter
            continue
        seen.add(a); stmts.append(node)
    if not stmts:
        return code, 0
    # Dependency ordering; only the order of existing asserts changes.
    #   0 type checks, no subscript  -> must come first, or the rest crash with TypeError
    #   1 checks without element access (len, comparisons)
    #   2 subscript access (x[i])    -> needs the length check before it, or IndexError
    def _rank(node):
        s = ast.unparse(node)
        has_sub = any(isinstance(x, ast.Subscript) for x in ast.walk(node))
        if has_sub:
            return 2
        return 0 if ("isinstance" in s or "type(" in s) else 1
    stmts = sorted(stmts, key=_rank)
    ins = 1 if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)) else 0
    fn.body[ins:ins] = stmts
    return ast.unparse(tree), len(stmts)


def fn_of(tree, entry):
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for f in fns:
        if f.name == entry:
            return f
    return fns[0] if fns else None


def attach(code, tid, entry):
    spec = specs().get(tid) or {"specs": [], "unparsed": []}
    ss = sorted(spec["specs"], key=lambda s: ORDER.get(s["kind"], 3))
    if not ss:
        return code, 0, len(spec["unparsed"])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, -1, len(spec["unparsed"])
    fn = fn_of(tree, entry)
    if fn is None:
        return code, -1, len(spec["unparsed"])
    fn_args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.posonlyargs} \
              | {a.arg for a in fn.args.kwonlyargs}
    stmts, seen = [], set()
    for s in ss:
        a = spec_to_assert(s)
        if not a or a in seen:
            continue
        # the referenced argument must exist in the signature
        ref = s.get("arg") or s.get("subject", "")
        base_arg = ref.replace("len(", "").rstrip(")")
        if base_arg and base_arg not in fn_args:
            continue
        try:
            node = ast.parse(a).body[0]
        except SyntaxError:
            continue
        seen.add(a); stmts.append(node)
    if not stmts:
        return code, 0, len(spec["unparsed"])
    # insert after the docstring
    ins = 1 if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)) else 0
    fn.body[ins:ins] = stmts
    return ast.unparse(tree), len(stmts), len(spec["unparsed"])


def main(in_path, out_path):
    recs = json.load(open(in_path))
    n_ast_fail = 0; n_attached = 0; total_asserts = 0; total_unparsed = 0
    out = []
    for r in recs:
        tid = str(r["id"])
        code, n, un = attach(r.get("code") or "", tid, r.get("entry_point") or "")
        rec = dict(r)
        rec.update(code=code, armA_asserts=n, armA_unparsed_clauses=un)
        if n == -1:
            n_ast_fail += 1
        elif n > 0:
            n_attached += 1; total_asserts += n
        total_unparsed += un
        out.append(rec)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[template] {len(out)} tasks | asserts inserted in {n_attached} tasks "
          f"({total_asserts} total) | AST failures kept as-is {n_ast_fail} | "
          f"unparsed conditions {total_unparsed} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
