"""Segmentation and lexical primitives shared by the specification graph.

A description line is split into clause-like segments; each segment is summarised by its
content-word set (`cw`), tested for functional-role markers (`FUNC_MARK`, `NUMBERED`) and for
verbatim regions that must never be touched (`CODE_LINE`, doctest lines, code). `repair` cleans the
seam a removal leaves behind. `anchor_mask` is the lexical-only masker kept as a comparison arm;
the method's masker is utils/functional_view.py, which adds the embedding condition.

Nothing here guesses whether a sentence "looks like" a contract: a segment is removed only when it
lexically anchors to an extracted condition, and functional segments are guarded even when they
anchor. When in doubt the segment stays — leaving a condition in the view is cheaper than
destroying a functional requirement.
"""
import re, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "code")

STOP = {"the","a","an","of","be","is","are","and","or","to","in","for","that","this","it","with",
        "on","at","by","as","not","no","must","should","if","all","any","each","when","then",
        "than","has","have","had","was","were","will","can","may","might","its","their","they",
        "there","these","those","from","into","only","also","but","we","you","your","i",
        "function","input","inputs","value","values","given","which"}
FUNC_MARK = re.compile(r"\b(return|returns|goal|example|examples|valid\s+if|rules?|format|"
                       r"output|task|write|compute|find|separate|>>>)\b", re.I)
NUMBERED = re.compile(r"^\s*(\d+[.)]|[*\-•])\s")
CODE_LINE = re.compile(r'^\s*(>>>|assert|def |from |import |#)')
"""Lines that are code rather than prose: a doctest call, an assert, the signature, an import, a
comment. Matched at the START of a line and taken out before segmentation, so they are never split
and never removed.

The name is deliberate. Of the 492 lines this catches in the benchmark, 140 are signature, import
or comment rather than examples, so calling it an example rule would misdescribe it. Recovering
executable examples is a separate path (utils/example_signals.py) that re-scans the same text with
its own parsers; the two share no code, only the textual conventions they happen to look for.
"""
DOCSTR_OPEN = re.compile(r'^(\s*(?:"""|\'\'\'))\s*(\S.*)$')   # docstring marker + inline content


def cw(text):
    """content-word keys (crude 6-char stem, stopword-removed)."""
    return {w[:6] for w in re.findall(r"[a-z][a-z0-9\-']*", text.lower()) if w not in STOP}


def segments(line):
    """clause-ish segments: split on sentence/semicolon boundaries (punct kept with left part)."""
    parts = re.split(r"(?<=[.;])\s+", line)
    return [p for p in parts if p.strip()]


def repair(line):
    """seam cleanup after segment removal (rule pass; CFG-parser upgrade point)."""
    s = re.sub(r"\s+([.,;:])", r"\1", line)
    s = re.sub(r"([.;,:])[\s]*(?:[.;,]+)", r"\1", s)      # dup punctuation
    s = re.sub(r"^\s*[;,]\s*", "", s)                     # leading orphan
    s = re.sub(r";\s*$", ".", s)                          # dangling semicolon
    s = re.sub(r";\s+(?=[A-Z])", ". ", s)                 # ';' before new sentence -> '.'
    s = re.sub(r"\s{2,}", " ", s).rstrip()
    return s


def anchor_mask(prompt_cs, contract_nl, thr=0.55):
    """Remove only segments that lexically anchor to contract_nl; guard functional segments."""
    ck = cw(contract_nl)
    removed, kept_lines = [], []
    for ln in prompt_cs.split("\n"):
        if CODE_LINE.match(ln) or NUMBERED.match(ln) or not ln.strip():
            kept_lines.append(ln); continue
        prefix = ""
        m = DOCSTR_OPEN.match(ln)
        if m:                                   # '""" text...' -> protect marker, process text
            prefix, ln = m.group(1) + " ", m.group(2)
        elif ln.strip() in ('"""', "'''"):
            kept_lines.append(ln); continue
        keep = []
        for seg in segments(ln):
            sk = cw(seg)
            ov = len(sk & ck) / max(1, len(sk))
            if len(sk) >= 3 and ov >= thr and not FUNC_MARK.search(seg):
                removed.append(seg.strip()); continue
            keep.append(seg)
        if keep:
            newln = repair(" ".join(keep))
            indent = ln[:len(ln) - len(ln.lstrip())]
            kept_lines.append((prefix or indent) + newln if newln else prefix + ln)
        elif prefix:                            # all content removed but docstring marker must stay
            kept_lines.append(prefix.rstrip())
        # line fully removed -> drop
    return "\n".join(kept_lines), removed


# ---- metrics ----

SEAM_PAT = [re.compile(r";\s*[A-Z]"), re.compile(r",\s*\."), re.compile(r"\.\s+[a-z]"),
            re.compile(r"\b(and|or|but)\s*[.;]"), re.compile(r":\s*\.")]


def seam_errors(orig, masked):
    """seam patterns present in masked but not (as many) in original."""
    n = 0
    for p in SEAM_PAT:
        if len(p.findall(masked)) > len(p.findall(orig)):
            n += 1
    return n


def doctest_kept(orig, masked):
    o = [l.strip() for l in orig.split("\n") if l.strip().startswith(">>>")]
    m = set(l.strip() for l in masked.split("\n"))
    return all(l in m for l in o)


def func_seg_retention(orig, masked):
    """fraction of functional-marker segments of orig retained (substring) in masked."""
    fsegs = [s for ln in orig.split("\n") for s in segments(ln)
             if FUNC_MARK.search(s) and not ln.strip().startswith(">>>")]
    if not fsegs:
        return 1.0
    return sum(1 for s in fsegs if s.strip()[:60] in masked) / len(fsegs)


def contract_removal(masked, contract_nl):
    """fraction of contract-anchored segments still present in masked (lower=better removal).
    returns removed-rate = 1 - present."""
    ck = cw(contract_nl)
    csegs = [s for ln in masked.split("\n") for s in segments(ln)
             if len(cw(s)) >= 3 and len(cw(s) & ck) / max(1, len(cw(s))) >= 0.55]
    return len(csegs)  # residual anchored segments in output
