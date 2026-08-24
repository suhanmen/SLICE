"""Positional argument renaming, then re-insertion of the assertions.

The conditions are extracted with the dataset signature in view, so they name the arguments as the
signature does; the body is generated without that anchor and may use different names. Since the
tests call positionally, the i-th signature parameter is the i-th body parameter, which makes a
positional rename deterministic and safe. Without it, `insert_asserts` correctly drops every
assertion as referencing nothing.

    python code/utils/rename_and_insert.py <assertions.json> <selected.json> <out.json>
"""
import json, sys, os, ast, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from utils.attach_assertions import insert_asserts, fn_of

TASKS = {str(t["id"]): t for t in json.load(open("dataset/contracteval/test.json"))}


def sig_params(sig):
    m = re.search(r"\(([^)]*)\)", sig or "")
    if not m: return []
    return [a.split("=")[0].split(":")[0].strip() for a in m.group(1).split(",") if a.strip()]


class Rename(ast.NodeTransformer):
    def __init__(self, mapping): self.m = mapping
    def visit_Name(self, node):
        if node.id in self.m: node.id = self.m[node.id]
        return node


def main(arm_path, sel_path, out_path):
    arm = json.load(open(arm_path))
    sel = {str(r["id"]): r for r in json.load(open(sel_path))}
    n_fix = n_already = n_fail = 0
    out = []
    for r in arm:
        tid = str(r["id"])
        rec = dict(r)
        body = (sel.get(tid) or {}).get("code") or ""
        raw = r.get("raw_assertions") or ""
        lines = [l.strip().lstrip("0123456789. ") for l in raw.split("\n")]
        asserts = [l for l in lines if l.startswith("assert")]
        if not asserts or not body:
            out.append(rec); continue
        try:
            fn = fn_of(ast.parse(body), r.get("entry_point") or "")
        except SyntaxError:
            out.append(rec); continue
        if fn is None:
            out.append(rec); continue
        bparams = [a.arg for a in fn.args.args]
        sparams = sig_params(TASKS.get(tid, {}).get("signature"))
        mapping = {s: b for s, b in zip(sparams, bparams) if s != b}
        fixed = []
        for a in asserts:
            try:
                node = ast.parse(a).body[0]
            except SyntaxError:
                continue
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if not (names & set(bparams)) and mapping:      # references no body parameter -> remap
                node = Rename(mapping).visit(node)
            fixed.append(ast.unparse(node))
        code, n = insert_asserts(body, r.get("entry_point") or "", fixed)
        prev = r.get("n_asserts") or 0
        if n > prev: n_fix += 1
        elif n == prev and n > 0: n_already += 1
        elif n <= 0: n_fail += 1
        rec.update(code=code, n_asserts=n)
        out.append(rec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[remap] {len(out)} tasks | recovered by renaming {n_fix} | unchanged {n_already} | "
          f"still zero {n_fail} -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
