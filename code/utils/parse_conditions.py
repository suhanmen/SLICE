"""Parse each condition into a structured specification (the template comparison arm).

Input  : the input-precondition nodes of artifacts/condition_nodes.json
Output : the path given as argv[1] (artifacts/condition_specs_new.json by default)
         {task id: {"specs": [{clause, kind, arg, ...}], "unparsed": [clause, ...]}}

kinds: type {types} | range {subject, op, bound} | nonempty {arg} | membership {arg, allowed}
A condition the rules cannot parse goes to `unparsed`, i.e. it is left to the LLM arm.

Only the condition text and the parameter names of the signature are consumed.

NOTE: the shipped artifacts/condition_specs.json is the frozen file the reported runs used. It
predates the relaxation of condition_decompose.py that accepts conditions not naming an argument,
so re-running this parser yields a strict superset (744 specs over 845 conditions vs the frozen 719
over 766; shared conditions are classified identically). It therefore writes to a path given on the
command line rather than overwriting the frozen file.
"""
import json, re, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

_NODES = None


def nodes():
    global _NODES
    if _NODES is None:
        p = "artifacts/condition_nodes.json"
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} is missing — run scripts/build_artifacts.sh to rebuild it")
        _NODES = {t["id"]: t for t in json.load(open(p, encoding="utf-8"))}
    return _NODES

TYPE_WORDS = {
    "integer": "int", "int": "int", "string": "str", "str": "str", "float": "float",
    "list": "list", "array": "list", "tuple": "tuple", "dict": "dict",
    "dictionary": "dict", "boolean": "bool", "bool": "bool", "number": "(int, float)",
    "numeric": "(int, float)", "set": "set",
}
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}


def num_of(tok):
    tok = tok.strip().rstrip(".,;")
    if tok.lstrip("-").replace(".", "", 1).isdigit():
        return float(tok) if "." in tok else int(tok)
    return NUM_WORDS.get(tok.lower())


def find_arg(text, args):
    """Which argument the condition refers to: an explicit name, then an ordinal, then
    the single parameter of a one-argument function."""
    for a in args:
        if re.search(rf"(?<![\w'\"]){re.escape(a)}(?![\w'\"])", text):
            return a
    ordinals = ["first", "second", "third", "fourth"]
    for i, w in enumerate(ordinals):
        if re.search(rf"\b{w}\b", text, re.I) and i < len(args):
            return args[i]
    if re.search(r"\b(the input|input)\b", text, re.I) and args:
        return args[0]          # a singular "the input" means the first parameter
    if re.search(r"\b(the list|the array|the string|the value|the variable)\b", text, re.I) \
            and len(args) == 1:
        return args[0]
    if len(args) == 1:
        return args[0]
    return None


def parse_clause(text, args):
    t = " ".join(text.split())
    low = t.lower()
    arg = find_arg(t, args)
    specs = []

    # --- membership: "each character ... either A, B, or C" / "contain only ..."
    if re.search(r"\b(each (character|char|element)|contain only|only contain|only the characters|"
                 r"consists?\s+(?:only\s+)?of)\b", low):
        chars = re.findall(r"'([^']*)'|\"([^\"]*)\"", t)
        allowed = [a or b for a, b in chars]
        # spelled-out character names
        for name, ch in [("opening parenthesis", "("), ("closing parenthesis", ")"),
                         ("open and close parentheses", "()"), ("parentheses", "()"),
                         ("space", " "), ("comma", ","), ("digit", "CLASS_DIGIT"),
                         ("letter", "CLASS_ALPHA"), ("alphabetic", "CLASS_ALPHA")]:
            if name in low:
                for c in (ch if ch.startswith("CLASS") else list(ch)):
                    cc = ch if ch.startswith("CLASS") else c
                    if cc not in allowed:
                        allowed.append(cc)
        if allowed and arg:
            specs.append({"kind": "membership", "arg": arg,
                          "allowed": list(dict.fromkeys(allowed)), "element": "char"})

    # --- type: "must be of type X" / "must be an X" / "X or Y"
    m = re.findall(r"\b(?:of type|be an?|be of type|are|be)\s+((?:integer|int|string|str|float|list|array|tuple|dict|dictionary|boolean|bool|number|numeric|set)s?)"
                   r"(?:\s+or\s+((?:integer|int|string|str|float|number)s?))?", low)
    if m:
        types = []
        for t1, t2 in m:
            for w in (t1, t2):
                w = w.rstrip("s") if w and w not in ("s",) else w
                if w and TYPE_WORDS.get(w) and TYPE_WORDS[w] not in types:
                    types.append(TYPE_WORDS[w])
        if types and arg:
            specs.append({"kind": "type", "arg": arg, "types": types})

    # --- element-type: "all elements in the list must be integers"
    m2 = re.search(r"\b(?:all|each)\s+(?:the\s+)?elements?\b.*?\bbe\s+((?:integer|int|string|str|float|number|numeric)s?)"
                   r"(?:\s+or\s+((?:integer|int|string|str|float|number)s?))?", low)
    if m2 and arg:
        types = [TYPE_WORDS[w.rstrip("s")] for w in m2.groups() if w and TYPE_WORDS.get(w.rstrip("s"))]
        if types:
            specs.append({"kind": "elements_type", "arg": arg, "types": list(dict.fromkeys(types))})

    # --- nonempty
    if re.search(r"\bnon-?empty\b|\bnot (be )?empty\b|\bis not empty\b", low) and arg:
        specs.append({"kind": "nonempty", "arg": arg})

    # --- exact count: "contain exactly two elements"
    m3 = re.search(r"\bexactly\s+([\w]+)\s+elements?\b", low)
    if m3 and arg:
        n = num_of(m3.group(1))
        if n is not None:
            specs.append({"kind": "range", "subject": f"len({arg})", "op": "==", "bound": n})

    # --- range, including a len() subject
    subject = f"len({arg})" if (arg and re.search(r"\b(length|len)\b", low)) else arg
    # between A and B (inclusive)
    m4 = re.search(r"between\s+([\w.\-]+)\s+and\s+([\w.\-]+)", low)
    if m4 and subject:
        lo, hi = num_of(m4.group(1)), num_of(m4.group(2))
        if lo is not None:
            specs.append({"kind": "range", "subject": subject, "op": ">=", "bound": lo})
        if hi is not None:
            specs.append({"kind": "range", "subject": subject, "op": "<=", "bound": hi})
    for pat, op in [
        (r"greater than or equal to\s+([\w.\-]+)", ">="),
        (r"less than or equal to\s+([\w.\-]+)", "<="),
        (r"at least\s+([\w.\-]+)", ">="),
        (r"at most\s+([\w.\-]+)", "<="),
        (r"greater than\s+([\w.\-]+)", ">"),
        (r"less than\s+([\w.\-]+)", "<"),
        (r"equal to\s+([\w.\-]+)", "=="),
    ]:
        for tok in re.findall(pat, low):
            n = num_of(tok)
            if n is not None and subject:
                specs.append({"kind": "range", "subject": subject, "op": op, "bound": n})
    if not any(s["kind"] == "range" for s in specs):
        if re.search(r"\bnon-?negative\b", low) and subject:
            specs.append({"kind": "range", "subject": subject, "op": ">=", "bound": 0})
        elif re.search(r"\bpositive\b", low) and subject:
            specs.append({"kind": "range", "subject": subject, "op": ">", "bound": 0})
        elif re.search(r"\bnegative\b", low) and subject:
            specs.append({"kind": "range", "subject": subject, "op": "<", "bound": 0})

    # dedup
    seen, out = set(), []
    for s in specs:
        k = json.dumps(s, sort_keys=True)
        if k not in seen:
            seen.add(k); out.append(s)
    return out


def main():
    result, tot, parsed_n, unparsed_n = {}, 0, 0, 0
    kind_count = {}
    for tid, t in nodes().items():
        args = t.get("args") or []
        specs, unparsed = [], []
        for c in t["contract_nodes"]:
            if not c.get("is_input_precondition"):
                continue
            tot += 1
            ps = parse_clause(c["text"], args)
            if ps:
                parsed_n += 1
                for p in ps:
                    p["clause"] = c["text"]
                    kind_count[p["kind"]] = kind_count.get(p["kind"], 0) + 1
                specs.extend(ps)
            else:
                unparsed_n += 1
                unparsed.append(c["text"])
        result[tid] = {"specs": specs, "unparsed": unparsed}
    out = sys.argv[1] if len(sys.argv) > 1 else "artifacts/condition_specs_new.json"
    json.dump(result, open(out, "w"), indent=1)
    print(f"input-precondition clauses: {tot}")
    print(f"parsed: {parsed_n} ({parsed_n/max(tot,1)*100:.1f}%)  | kinds: {kind_count}")
    print(f"unparsed, left to the LLM arm: {unparsed_n}")
    un = [u for v in result.values() for u in v["unparsed"]]
    print("\nunparsed examples (10):")
    for u in un[:10]:
        print("  -", u[:90])


if __name__ == "__main__":
    main()
