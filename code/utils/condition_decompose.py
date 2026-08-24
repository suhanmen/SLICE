"""Rule-based decomposition of a specification into atomic nodes.

Splits contract text into individual conditions (`decompose_contract`, used by the specification
graph and by stage (iii)) and a description into behaviour vs example nodes (`decompose_func`).
Running the module also writes a quality report: how many conditions each rule finds, how the
count compares with the reference clause count, and how often a precondition sentence leaks into
the behaviour nodes.

The rules follow what the corpus actually looks like: contract text is highly regular
("<subject> must be <predicate>"), some sentences are runtime invariants rather than input
preconditions ("... at all times", "when ..."), and descriptions mix prose with examples
(f(...) = ..., >>>, a trailing literal).

Writes artifacts/{condition_nodes.json, split_report.md}
"""
import json, re, sys, os, ast
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "dataset/contracteval/test.json")
OUTDIR = os.path.join(ROOT, "artifacts")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- helpers ----------

def arg_names(signature):
    """Extract parameter names from a `def f(...)` signature string."""
    try:
        tree = ast.parse(signature.strip().rstrip(":") + ":\n    pass")
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
        if not fn:
            return []
        return [a.arg for a in fn.args.args] + [a.arg for a in fn.args.posonlyargs] + \
               [a.arg for a in fn.args.kwonlyargs]
    except SyntaxError:
        return []


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------- contract decomposition ----------

RUNTIME_MARKERS = re.compile(r"\b(at all times|when a|when the|during|after each|each step|iteration)\b", re.I)
TYPE_PAT = re.compile(r"\b(of type|be an? (integer|int|string|str|float|number|list|dict|tuple|bool|boolean|array)|be (integers|strings|floats|numbers|lists))\b", re.I)
RANGE_PAT = re.compile(r"\b(greater than|less than|at least|at most|non-negative|nonnegative|positive|negative|>=|<=|>|<|equal to|greater than or equal|zero|within the range|in the range|between)\b", re.I)
MEMBER_PAT = re.compile(r"\b(either|one of|must be either|contain only|only the characters|consist of)\b", re.I)


def classify_contract(sent):
    if MEMBER_PAT.search(sent):
        return "membership"
    if TYPE_PAT.search(sent):
        return "type"
    if RANGE_PAT.search(sent):
        return "range"
    return "other"


def expand_subjects(sent, args):
    """'Both x and y must be integers' -> ['x must ...', 'y must ...']. Else single."""
    m = re.match(r"\s*Both\s+(\w+)\s+and\s+(\w+)\s+(.*)", sent, re.I)
    if m:
        a, b, rest = m.group(1), m.group(2), m.group(3)
        return [f"{a} {rest}", f"{b} {rest}"]
    return [sent]


def decompose_contract(contract_nl, args):
    nodes = []
    for sent in split_sentences(contract_nl):
        low = sent.lower()
        if "must" not in low and "should" not in low and "cannot" not in low and "can not" not in low:
            # not a constraint sentence
            if not re.search(r"\b(non-negative|positive|integer|string|greater|less)\b", low):
                continue
        for sub in expand_subjects(sent, args):
            is_runtime = bool(RUNTIME_MARKERS.search(sub))
            # Anything that is not a runtime invariant counts as an input precondition.
            # Requiring the sentence to name an argument dropped real preconditions that refer to
            # the arguments indirectly ("Both inputs ...", "The radius ...").
            nodes.append({
                "text": sub.strip(),
                "category": classify_contract(sub),
                "is_input_precondition": not is_runtime,
                "runtime_invariant": is_runtime,
            })
    return nodes


# ---------- func decomposition ----------

EXAMPLE_PAT = re.compile(
    r"(>>>.*)|([A-Za-z_]\w*\s*\([^)]*\)\s*(=|=>|->|returns?)\s*\S+)|(\[[^\]]*\]\s*$)|(\"0b[01]+\")"
)
EXAMPLE_LEADIN = re.compile(r"\b(for example|examples?|e\.g\.|such as)\b\s*:?", re.I)


def decompose_func(description):
    text = re.sub(r"\s+", " ", description.strip())
    # cut off an explicit example lead-in tail
    lead = EXAMPLE_LEADIN.search(text)
    example_span = ""
    if lead:
        example_span = text[lead.start():]
        text = text[:lead.start()].strip()
    behavior, examples = [], []
    if example_span:
        examples.append(example_span.strip())
    for sent in split_sentences(text):
        if EXAMPLE_PAT.search(sent):
            examples.append(sent)
        else:
            behavior.append(sent)
    # trailing bare list/example literal often glued to last behavior sentence
    return [{"text": b} for b in behavior if b], [{"text": e} for e in examples if e]


# ---------- run ----------

def main():
    data = json.load(open(DATA))
    out = []
    for r in data:
        args = arg_names(r["signature"])
        cnodes = decompose_contract(r["contract_nl"], args)
        bnodes, enodes = decompose_func(r["description"])
        gold = len(set(r.get("cvt_keys") or []))
        out.append({
            "id": r["id"], "entry_point": r["entry_point"], "args": args,
            "gold_clause_count": gold,
            "contract_nodes": cnodes,
            "input_precond_count": sum(1 for n in cnodes if n["is_input_precondition"]),
            "func_behavior_nodes": bnodes,
            "func_example_nodes": enodes,
        })
    json.dump(out, open(os.path.join(OUTDIR, "condition_nodes.json"), "w"), indent=2)

    # ---- validation report ----
    n = len(out)
    cat = Counter(c["category"] for t in out for c in t["contract_nodes"])
    runtime = sum(1 for t in out for c in t["contract_nodes"] if c["runtime_invariant"])
    total_c = sum(len(t["contract_nodes"]) for t in out)
    no_contract = sum(1 for t in out if not t["contract_nodes"])
    no_func = sum(1 for t in out if not t["func_behavior_nodes"])
    # Q2: input-precond count vs gold
    pairs = [(t["input_precond_count"], t["gold_clause_count"]) for t in out]
    exact = sum(1 for a, g in pairs if a == g)
    over = sum(1 for a, g in pairs if a > g)
    under = sum(1 for a, g in pairs if a < g)
    mad = sum(abs(a - g) for a, g in pairs) / n
    # crude Pearson
    import statistics as st
    xs = [a for a, _ in pairs]; ys = [g for _, g in pairs]
    try:
        r_corr = st.correlation(xs, ys)
    except Exception:
        r_corr = float("nan")
    # cross-contamination proxy: 'must be' precondition-like sentence living in func behavior
    contam = sum(1 for t in out for b in t["func_behavior_nodes"]
                 if re.search(r"\bmust be\b", b["text"], re.I))
    func_beh = sum(len(t["func_behavior_nodes"]) for t in out)
    with_examples = sum(1 for t in out if t["func_example_nodes"])

    lines = []
    lines.append("# Rule-based spec decomposition — quality report\n")
    lines.append(f"- tasks: {n}\n")
    lines.append("## 1. Contract decomposition\n")
    lines.append(f"- contract nodes: {total_c} ({total_c/n:.2f} per task)")
    lines.append(f"- by category: {dict(cat)}")
    lines.append(f"- flagged as runtime invariant, not an input precondition: {runtime} "
                 f"({runtime/max(total_c,1)*100:.1f}%)")
    lines.append(f"- tasks with no contract node: {no_contract}\n")
    lines.append("## 2. Condition count vs the reference clause count\n")
    lines.append(f"- exact match: {exact}/{n} ({exact/n*100:.1f}%)")
    lines.append(f"- over-count: {over}  | under-count: {under}")
    lines.append(f"- mean abs diff: {mad:.2f}  | Pearson r: {r_corr:.3f}\n")
    lines.append("## 3. Description decomposition\n")
    lines.append(f"- behaviour nodes: {func_beh} ({func_beh/n:.2f} per task)")
    lines.append(f"- tasks with an example node: {with_examples}/{n} ({with_examples/n*100:.1f}%)")
    lines.append(f"- tasks with no behaviour node: {no_func}\n")
    lines.append("## 4. Cross-contamination (proxy for a failed split)\n")
    lines.append(f"- 'must be' precondition sentences left in the behaviour nodes: {contam} "
                 f"of {func_beh} ({contam/max(func_beh,1)*100:.1f}%)\n")
    lines.append("## 5. First three tasks\n")
    for t in out[:3]:
        lines.append(f"### {t['id']} ({t['entry_point']}) — gold={t['gold_clause_count']}, "
                     f"input_precond={t['input_precond_count']}")
        lines.append("- contract nodes:")
        for c in t["contract_nodes"]:
            tag = "RUNTIME" if c["runtime_invariant"] else ("INPUT" if c["is_input_precondition"] else "other")
            lines.append(f"    - [{c['category']}/{tag}] {c['text']}")
        lines.append("- behaviour nodes:")
        for b in t["func_behavior_nodes"]:
            lines.append(f"    - {b['text']}")
        lines.append("- example nodes:")
        for e in t["func_example_nodes"]:
            lines.append(f"    - {e['text']}")
        lines.append("")
    open(os.path.join(OUTDIR, "split_report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {OUTDIR}/condition_nodes.json , split_report.md")


if __name__ == "__main__":
    main()
