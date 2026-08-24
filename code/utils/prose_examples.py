"""Recover example checks written in prose rather than as doctests (deterministic, no model).

Finds `f(args) ==|=>|==>|->|returns y` inside the description text and turns it into an assert.
Both the arguments and the expected value must be literals, so the check runs without context.
This is the fallback for the tasks whose examples example_signals.py cannot parse.

Writes artifacts/prose_examples.json as {task id: [assert, ...]}
"""
import json, os, re, ast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SEP_RE = r"(?:==>|=>|➞|->|==|should\s+return|returns?\s*:?)"


def balanced_call(text, start, ep):
    """Return the call expression starting at `start`, up to the balanced closing paren."""
    i = text.index("(", start)
    depth = 0
    for j in range(i, len(text)):
        ch = text[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1], j + 1
    return None, None


def literal_ok(expr):
    try:
        ast.literal_eval(expr)
        return True
    except Exception:
        return False


def call_is_literal(call):
    """True if every argument is a literal, so the call runs without any context."""
    try:
        node = ast.parse(call, mode="eval").body
        if not isinstance(node, ast.Call) or node.keywords:
            return False
        return all(literal_ok(ast.unparse(a)) for a in node.args)
    except Exception:
        return False


def safe_arith(expr):
    """Fold an arithmetic expression of numeric constants ("19 - 5 - 6") to its value."""
    try:
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                                  ast.Mod, ast.Pow, ast.USub, ast.UAdd)):
                return None
            if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
                return None
        return repr(eval(compile(node, "<arith>", "eval"), {"__builtins__": {}}))
    except Exception:
        return None


def parse_rhs(rest, ep):
    """Parse the literal expected value from the text after the separator."""
    cand = rest.strip().splitlines()[0].strip()
    # prose often chains examples on one line; cut before the next call
    nxt = re.search(re.escape(ep) + r"\s*\(", cand)
    if nxt:
        cand = cand[:nxt.start()]
    for _ in range(10):
        cand = cand.strip()
        if not cand:
            return None
        if literal_ok(cand):
            return cand
        # lowercase true/false
        if re.fullmatch(r"(?i)true|false", cand):
            return cand.capitalize()
        # "19 - 5 - 6 = 8": take the final value
        if "=" in cand and "==" not in cand:
            tail = cand.rsplit("=", 1)[1].strip()
            if literal_ok(tail):
                return tail
            cand = cand.rsplit("=", 1)[0].strip()
            continue
        folded = safe_arith(cand)
        if folded is not None:
            return folded
        stripped = cand.rstrip(" .,;:)")
        if stripped != cand:
            cand = stripped
            continue
        cand2 = re.sub(r"\s+(?:and|or|because|since|as|which|so)\b.*$", "", cand)
        cand = cand2 if cand2 != cand else cand[:-1]
    return None


def extract(desc, ep):
    out = []
    for m in re.finditer(re.escape(ep) + r"\s*\(", desc):
        call, end = balanced_call(desc, m.start(), ep)
        if not call or not call_is_literal(call):
            continue
        sep = re.match(r"\s*" + SEP_RE, desc[end:])
        if not sep:
            continue
        rhs = parse_rhs(desc[end + sep.end():], ep)
        if rhs is None:
            continue
        stmt = f"assert {call} == {rhs}"
        try:
            ast.parse(stmt)
        except Exception:
            continue
        if stmt not in out:
            out.append(stmt)
    return out


def main():
    # The tasks that need this: in the frozen set, but with no check example_signals could parse.
    sig = json.load(open("artifacts/example_signals.json"))
    frozen = json.load(open("dataset/contracteval/eval_tasks_340.json"))["ids"]
    zone = [t for t in frozen if not (sig.get(t) or {}).get("checks")]
    data = {t["id"]: t for t in json.load(open("dataset/contracteval/test.json"))}
    res = {}
    for tid in zone:
        t = data[tid]
        ex = extract(t.get("description", ""), t.get("entry_point", ""))
        if ex:
            res[tid] = ex[:5]
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/prose_examples.json"
    json.dump(res, open(path, "w"), indent=1)
    n_as = sum(len(v) for v in res.values())
    print(f"[prose] recovered {len(res)}/{len(zone)} tasks, {n_as} asserts -> {path}")
    for tid in list(res)[:6]:
        print(f"--- {tid}:")
        for s in res[tid]:
            print("   ", s)


if __name__ == "__main__":
    main()
