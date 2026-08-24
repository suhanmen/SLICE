"""Rule-based contract masker: strip input-precondition clauses from a (cs) problem prompt so the
BODY sees a functionality-only description — derived from prompt_cs, WITHOUT using prompt_base
(deployment-realistic). A clause is a contract precondition if it has a modal marker AND a
type/constraint word and is NOT describing return/output behavior. Paired with body_constrain
(assert/raise masking) as an output-side backstop for anything the rule misses.

Validated (n=364): masked_cs is much closer to prompt_base than raw cs
(BLEU 0.49->0.61, difflib 0.75->0.86, ROUGE-L 0.70->0.81), i.e. it removes the contract.
"""
import re

_MARK = re.compile(r"\b(must|should|has\s+to|have\s+to|is\s+required|are\s+required|"
                   r"assume[ds]?|assumes|guaranteed|of\s+type)\b", re.I)
_TYPE = re.compile(r"\b(integer|string|list|float|number|positive|negative|non-?negative|"
                   r"non-?empty|of\s+type|tuple|dict|boolean|bool|char|digit|element|array|"
                   r"length|balanced|sorted|unique|contain|type\s+str|greater\s+than|less\s+than|"
                   r"at\s+least|at\s+most|>=|<=)\b", re.I)
_BEHAVIOR = re.compile(r"\b(return|returns|output|outputs|print|prints|raise|give|gives|"
                       r"be\s+returned|produce)\b", re.I)


# Functional cues: a clause about OUTPUT/behavior or a rule spec — NOT an input precondition.
_FUNCTIONAL = re.compile(r"\b(valid\s+if|is\s+valid|the\s+function|format|return|returns|output|"
                         r"should\s+return|rules?\b|following)\b", re.I)
_NUMBERED = re.compile(r"^\s*(\d+[.)]|[*\-•])\s")            # numbered/bulleted rule lines
_PARAMISH = re.compile(r'("\w+"|\b[a-z_]\w*\b)\s+(must|should|has\s+to|is\s+required|of\s+type)',
                       re.I)  # references a parameter name before the modal


def _is_contract(p: str) -> bool:
    if not _MARK.search(p) or not _TYPE.search(p):
        return False
    if _BEHAVIOR.search(p) or _FUNCTIONAL.search(p):   # return/output/valid-if/rules = functional
        return False
    return True     # marker + type word, not functional -> input precondition (drop)


def mask_contract(text: str) -> str:
    """Drop input-precondition clauses; keep functionality, rules, doctest examples, code.
    Safety: never drop numbered/bulleted rule lines; if a line loses >55% of its chars, keep the
    original (over-masking guard) so functional specs aren't destroyed."""
    if not text:
        return text
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith(">>>") or s.startswith("assert") or s.startswith("def ") \
           or s.startswith("from ") or s.startswith("import ") or s == '"""' or s.startswith("#") \
           or _NUMBERED.match(ln):
            out.append(ln); continue
        parts = re.split(r"(?<=[.;])\s+", ln)
        kept = [p for p in parts if not _is_contract(p)]
        rebuilt = " ".join(x for x in kept if x.strip())
        rebuilt = re.sub(r"\s*;\s*", "; ", rebuilt)
        rebuilt = re.sub(r"\s{2,}", " ", rebuilt).replace(" ;", ";").strip()
        # over-masking guard: if the line was gutted, keep the original prose line
        if rebuilt and len(rebuilt) < 0.45 * len(ln.strip()):
            out.append(ln); continue
        out.append(("    " + rebuilt) if (ln.startswith("    ") and rebuilt) else rebuilt)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))
