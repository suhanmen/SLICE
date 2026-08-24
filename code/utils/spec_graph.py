"""Specification graph construction and the graph-based mask.

    nodes  contract conditions (decomposed) | prompt_cs segments | examples (>>>, numbered lines)
    edges  anchor    condition <-> segment, when the segment lexically restates the condition
           entangle  segment  -> FUNC, when the segment carries a functional role
    rule   remove a segment iff it has an anchor edge and no entangle edge

Paper terminology: the `anchor` edge is the contract-grounding edge; an `entangled` segment is
one carrying a functional or example-preservation label. The internal names predate the paper.

`entangle` is deliberately not similarity-based. A contract restates the description's own
vocabulary, so a lexical criterion would protect exactly the segments that should be removed;
the edge therefore keys on behavioural-role cues (functional markers, numbered/bulleted lines).

After a removal, each kept sentence is driven through a small automaton (`dfa_state` /
`dfa_repair`): every terminal state has a defined action, so unlike a blacklist of patterns there
is no gap. A state with no repair transition (an unbalanced bracket or quote) undoes the removal
for that line.

    PYTHONPATH=code python code/utils/spec_graph.py

Writes artifacts/{spec_graphs.json, functional_view_base.json, graph_mask_report.md}
"""
import json, re, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)
sys.path.insert(0, "code")
from utils.segment_parser import (cw, segments, FUNC_MARK, NUMBERED, CODE_LINE, DOCSTR_OPEN,
                         seam_errors, doctest_kept, func_seg_retention, contract_removal,
                         anchor_mask as v1_mask)
from utils.condition_decompose import decompose_contract, arg_names
from utils.rule_mask_baseline import mask_contract as old_mask

# ---------------- SentenceDFA ----------------

CONJ = {"and", "or", "but", "because", "since", "while", "whereas", "however", "then"}
# Deontic markers. Without one, a lexical overlap is just narration and must not be removed.
MODAL = re.compile(r"\b(must|should|has\s+to|have\s+to|required|cannot|can\s+not|of\s+type|"
                   r"assumed?|guaranteed)\b", re.I)
DOCSTR_CLOSE = re.compile(r"\s*(\"\"\"|''')\s*$")   # trailing docstring marker: detach and keep


def dfa_state(sent):
    """Scan sentence, return final automaton state."""
    t = sent.strip()
    if not t:
        return "EMPTY"
    depth = 0; sq = 0; dq = 0
    for i, ch in enumerate(t):
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1
        elif ch == '"': dq ^= 1
        elif ch == "'":
            # an in-word apostrophe is not a quote
            prev_a = i > 0 and t[i-1].isalnum()
            next_a = i + 1 < len(t) and t[i+1].isalnum()
            if not (prev_a and next_a):
                sq ^= 1
    if depth != 0 or sq or dq:
        return "UNBALANCED"
    first = t.split()[0].rstrip(".,;:").lower()
    if t[0] in ";,." or (first in CONJ and t[0].islower()):
        return "ORPHAN_START"
    if re.search(r"\b(and|or|but)\s*[.;:]$", t):
        return "CONJ_TERM"
    if t.endswith((".", "?", "!", ":")):
        return "ACCEPT"
    if t.endswith(";"):
        return "SEMI"
    if t.endswith(","):
        return "COMMA"
    last = t.split()[-1].lower()
    if last in CONJ:
        return "CONJ_END"
    return "NO_TERM"


REPAIR = {   # transitions from a non-accepting state to ACCEPT
    "SEMI":       lambda s: s.rstrip()[:-1] + ".",
    "COMMA":      lambda s: s.rstrip()[:-1] + ".",
    "NO_TERM":    lambda s: s.rstrip() + ".",
    "CONJ_END":   lambda s: re.sub(r"\s+\w+$", "", s.rstrip()) + ".",
    "CONJ_TERM":  lambda s: re.sub(r"\s*\b(and|or|but)\s*([.;:])$", r"\2", s.rstrip()),
    "ORPHAN_START": lambda s: re.sub(r"^\s*(?:[;,.]+|\b(?:and|or|but|then)\b)\s*", "", s.strip()),
}


def dfa_repair(sent, max_steps=4):
    """Drive sentence to ACCEPT; None if unrecoverable (caller undoes the mask)."""
    s = sent
    for _ in range(max_steps):
        st = dfa_state(s)
        if st == "ACCEPT":
            return s
        fix = REPAIR.get(st)
        if fix is None:               # UNBALANCED / EMPTY -> no transition defined
            return None
        s2 = fix(s)
        if s2 == s:                   # transition made no progress
            return None
        s = s2
    return s if dfa_state(s) == "ACCEPT" else None


# ---------------- spec graph ----------------

def spec_text(task):
    """The condition text the graph anchors against.

    The method puts the model's own extracted conditions in `spec_text`. `contract_nl` (the
    dataset's reference contract) is the fallback, and is only ever reached by the comparison arms.
    """
    return task.get("spec_text") or task.get("contract_nl") or ""


def build_graph(task):
    """nodes: contract clauses / cs segments / examples; edges: anchor + entangle."""
    args = arg_names(task.get("signature") or "")
    cnodes = decompose_contract(spec_text(task), args)
    nodes, edges = [], []
    for i, c in enumerate(cnodes):
        nodes.append({"nid": f"c{i}", "kind": "contract", "category": c["category"],
                      "input_precond": c["is_input_precondition"], "text": c["text"]})
    ck = cw(spec_text(task))
    seg_id = 0
    for ln in (task.get("prompt_cs") or "").split("\n"):
        raw = ln
        m = DOCSTR_OPEN.match(ln)
        if m:
            ln = m.group(2)
        mc = DOCSTR_CLOSE.search(ln)
        if mc and ln[:mc.start()].strip():
            ln = ln[:mc.start()]                  # analyse the text without the closing marker
        if CODE_LINE.match(raw) or not ln.strip():
            if raw.strip():
                nodes.append({"nid": f"e{seg_id}", "kind": "code", "text": raw.strip()[:80]})
                seg_id += 1
            continue
        numbered = bool(NUMBERED.match(raw))
        for seg in segments(ln):
            nid = f"s{seg_id}"; seg_id += 1
            sk = cw(seg)
            ov = len(sk & ck) / max(1, len(sk))
            anchored = len(sk) >= 3 and ov >= 0.55 and bool(MODAL.search(seg))
            entangled = bool(FUNC_MARK.search(seg)) or numbered
            nodes.append({"nid": nid, "kind": "segment", "text": seg.strip()[:120],
                          "anchor_ov": round(ov, 2)})
            if anchored:
                # anchor edge -> best-matching contract node
                best, bi = 0.0, None
                for i, c in enumerate(cnodes):
                    o = len(cw(c["text"]) & sk) / max(1, len(sk))
                    if o > best:
                        best, bi = o, i
                edges.append({"src": f"c{bi if bi is not None else 0}", "dst": nid,
                              "type": "anchor", "w": round(ov, 2)})
            if entangled:
                edges.append({"src": nid, "dst": "FUNC", "type": "entangle"})
    return nodes, edges


def graph_mask(task):
    """remove segments with anchor-edge and no entangle-edge; DFA-repair; undo on failure."""
    nodes, edges = build_graph(task)
    anchor = {e["dst"] for e in edges if e["type"] == "anchor"}
    entangle = {e["src"] for e in edges if e["type"] == "entangle"}
    txt_of = {n["nid"]: n["text"] for n in nodes if n["kind"] == "segment"}
    removable_txt = {txt_of[n]: n for n in (anchor - entangle) if n in txt_of}

    removed, out_lines = [], []
    for ln in (task.get("prompt_cs") or "").split("\n"):
        raw = ln; prefix = ""; closer = ""
        m = DOCSTR_OPEN.match(ln)
        if m:
            prefix, ln = m.group(1) + " ", m.group(2)
        elif CODE_LINE.match(raw) or NUMBERED.match(raw) or not ln.strip():
            out_lines.append(raw); continue
        mc = DOCSTR_CLOSE.search(ln)
        if mc and ln[:mc.start()].strip():
            closer, ln = " " + mc.group(1), ln[:mc.start()]   # keep the closing marker
        keep, dropped = [], []
        for seg in segments(ln):
            if seg.strip()[:120] in removable_txt:
                dropped.append(seg.strip())
            else:
                keep.append(seg)
        if not dropped:                       # untouched line
            out_lines.append(raw); continue
        # DFA-repair each kept sentence; any failure -> undo this line
        repaired, ok = [], True
        for seg in keep:
            r = dfa_repair(seg)
            if r is None:
                ok = False; break
            repaired.append(r)
        if not ok:
            out_lines.append(raw); continue   # undo (do-no-harm)
        removed.extend(dropped)
        indent = raw[:len(raw) - len(raw.lstrip())]
        newln = " ".join(repaired).strip()
        if newln:
            out_lines.append((prefix or indent) + newln + closer)
        elif prefix or closer:
            out_lines.append((prefix + closer.strip()).strip() or (indent + closer.strip()))
    # cross-line seam: a ';' that introduced the removed clause now precedes a new sentence
    if removed:
        for i in range(len(out_lines) - 1):
            cur = out_lines[i]
            if CODE_LINE.match(cur) or NUMBERED.match(cur):
                continue
            nxt = next((l for l in out_lines[i + 1:] if l.strip()), "")
            ns = nxt.strip().lstrip('"\' ')
            if cur.rstrip().endswith(";") and ns[:1].isupper():
                out_lines[i] = cur.rstrip()[:-1] + "."
    return "\n".join(out_lines), removed, nodes, edges


# ---------------- run + metrics ----------------

def main():
    data = json.load(open("dataset/contracteval/test.json"))
    graphs, dump = {}, {}
    agg = {k: Counter() for k in ("old", "v1", "v2")}
    fsum = {k: 0.0 for k in ("old", "v1", "v2")}
    n = 0
    for r in data:
        pcs, cnl = r.get("prompt_cs") or "", r.get("contract_nl") or ""
        if not pcs.strip():
            continue
        n += 1
        o = old_mask(pcs)
        v1, v1_removed = v1_mask(pcs, cnl)
        v2, removed, nodes, edges = graph_mask(r)
        agg["v1"]["removed"] += bool(v1_removed)
        agg["v2"]["removed"] += bool(removed)
        graphs[r["id"]] = {"nodes": nodes, "edges": edges}
        dump[r["id"]] = {"masked": v2, "removed_segments": removed}
        for tag, mtxt in (("old", o), ("v1", v1), ("v2", v2)):
            agg[tag]["doctest"] += doctest_kept(pcs, mtxt)
            fsum[tag] += func_seg_retention(pcs, mtxt)
            agg[tag]["seam"] += seam_errors(pcs, mtxt) > 0
            agg[tag]["resid"] += contract_removal(mtxt, cnl)
            agg[tag]["changed"] += (mtxt != pcs)

    L = [f"# graph mask vs the lexical anchor and rule-mask arms — {n} tasks",
         "(generation covers all tasks; task-level scoring uses the frozen 340)\n",
         "| metric | rule mask | lexical anchor | graph + automaton |", "|---|---|---|---|",
         f"| text changed | {agg['old']['changed']} | {agg['v1']['changed']} | {agg['v2']['changed']} |",
         f"| **segments actually removed** | - | {agg['v1']['removed']} | {agg['v2']['removed']} |",
         f"| doctests kept | {agg['old']['doctest']}/{n} | {agg['v1']['doctest']}/{n} | {agg['v2']['doctest']}/{n} |",
         f"| functional segments kept | {fsum['old']/n:.3f} | {fsum['v1']/n:.3f} | {fsum['v2']/n:.3f} |",
         f"| tasks with a broken seam | {agg['old']['seam']} | {agg['v1']['seam']} | {agg['v2']['seam']} |",
         f"| contract segments left in | {agg['old']['resid']} | {agg['v1']['resid']} | {agg['v2']['resid']} |"]
    gstat = Counter()
    for g in graphs.values():
        gstat["nodes"] += len(g["nodes"]); gstat["anchor"] += sum(e["type"] == "anchor" for e in g["edges"])
        gstat["entangle"] += sum(e["type"] == "entangle" for e in g["edges"])
    L.append(f"\ngraph totals: {gstat['nodes']} nodes / {gstat['anchor']} anchor / {gstat['entangle']} entangle")
    L.append("\n## Spot checks\n")
    d = {str(x["id"]): x for x in data}
    for tid in ["HumanEval/1", "HumanEval/124", "HumanEval/102"]:
        v2, removed, nodes, edges = graph_mask(d[tid])
        prot = [e["src"] for e in edges if e["type"] == "entangle"]
        L.append(f"### {tid}")
        L.append(f"- removed: {removed if removed else '(nothing)'}")
        L.append(f"- segments protected by an entangle edge: {len(set(prot))}")
        L.append(f"- result (first 260 chars): {' '.join(v2.split())[:260]}")
        L.append("")
    rep = "\n".join(L)
    outd = "artifacts"
    os.makedirs(outd, exist_ok=True)
    open(os.path.join(outd, "graph_mask_report.md"), "w").write(rep)
    json.dump(graphs, open(os.path.join(outd, "spec_graphs.json"), "w"), indent=1)
    json.dump(dump, open(os.path.join(outd, "functional_view_base.json"), "w"), indent=1)
    print(rep)
    print(f"\n[written] {outd}/spec_graphs.json , functional_view_base.json , graph_mask_report.md")


if __name__ == "__main__":
    main()
