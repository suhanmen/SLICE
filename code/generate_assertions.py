"""Stage (iii): generate the contract assertions for each task and insert them into the body.

The selected body and the list of contract conditions are shown together; the model returns one
assert line per condition, which is inserted at the function entry by AST (attach_assertions).
Conditions are presented as natural language (`--conditions nl`, the method) or as the rule
parser's JSON specification (`--conditions json`, ablation). No gold code or test is read.

    python code/generate_assertions.py --model_name <hf_id> [--limit 340]

Writes output/<setting>/<model>/assertions_raw.json
"""
import argparse, json, sys, os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.attach_assertions import insert_asserts
from utils.gen_common import build_chat_prompt
from utils.paths import run_path, DEFAULT_SETTING

SYSTEM_FUNC = (
    "You are given a Python function body and its input-contract clauses.\n"
    "Write ONE standalone validator function that checks all the clauses with assert "
    "statements and returns nothing.\n"
    "Rules:\n"
    "- Name it validate_<function name>, taking the SAME parameters as the function.\n"
    "- Enforce exactly what the clauses state — nothing more, nothing less. Do NOT narrow "
    "types beyond the clauses.\n"
    "- Use the ACTUAL parameter names of the function as written in the code.\n"
    "- Look at how the function body uses each argument so the asserts do not reject inputs "
    "the body handles.\n"
    "- Output ONLY the validator function definition, no explanations, no code fences."
)

SYSTEM = (
    "You are given a Python function body and its input-contract clauses.\n"
    "Write exactly ONE assert statement per clause, to be placed at the top of the function.\n"
    "Rules:\n"
    "- Enforce exactly what each clause states — nothing more, nothing less.\n"
    "  Do NOT add checks that are not in the clauses. Do NOT narrow types beyond the clause.\n"
    "- Use the ACTUAL parameter names of the function as written in the code, even when a "
    "clause refers to arguments by different names (e.g. clause says 'x' but the function "
    "parameter is 'a' -> write the assert with 'a').\n"
    "- ORDER: put type checks (isinstance) FIRST, before any len()/comparison/membership "
    "assert — otherwise a wrong-typed input crashes with TypeError instead of being "
    "rejected by an assert.\n"
    "- When a clause states a type (e.g. 'must be a string'), ALWAYS include the isinstance "
    "check for it, even if the clause also has a length/range part.\n"
    "- Look at how the function body uses each argument so the asserts do not reject inputs "
    "the body handles.\n"
    "- Output ONLY the assert lines, one per line, no explanations, no code fences."
)


def compose_validator_func(body_code, entry, gen_text):
    """--vform func: prepend the generated validator function and call it at the body entry.
    Returns (new_code, #asserts in the validator); on failure (original code, 0 or -1)."""
    import ast as _ast, re as _re
    t = _re.sub(r"```(?:python)?|```", "", gen_text)
    # tolerate a missing "def" (e.g. the answer starts with "validate_f(x) -> None:")
    if not _re.search(r"^\s*def\s", t, _re.M) and _re.search(r"^\s*\w+\s*\(.*\)\s*(->[^:]+)?:", t, _re.M):
        t = _re.sub(r"^(\s*)(\w+\s*\(.*\)\s*(?:->[^:]+)?:)", r"\1def \2", t, count=1, flags=_re.M)
    try:
        vtree = _ast.parse(t)
    except SyntaxError:
        m = _re.search(r"^\s*def\s+\w+", t, _re.M)
        if not m: return body_code, -1
        try: vtree = _ast.parse(t[m.start():])
        except SyntaxError: return body_code, -1
    vfns = [n for n in vtree.body if isinstance(n, _ast.FunctionDef)]
    if not vfns: return body_code, -1
    vfn = vfns[0]
    n_asserts = sum(1 for x in _ast.walk(vfn) if isinstance(x, _ast.Assert))
    if n_asserts == 0: return body_code, 0
    try:
        btree = _ast.parse(body_code)
    except SyntaxError:
        return body_code, -1
    bfns = [n for n in _ast.walk(btree) if isinstance(n, _ast.FunctionDef) and n.name == entry] \
           or [n for n in _ast.walk(btree) if isinstance(n, _ast.FunctionDef)]
    if not bfns: return body_code, -1
    bfn = bfns[0]
    argnames = [a.arg for a in bfn.args.args]
    # call the validator with the body's own parameter names
    call = _ast.parse(f"{vfn.name}({', '.join(argnames)})").body[0]
    ins = 1 if (bfn.body and isinstance(bfn.body[0], _ast.Expr)
                and isinstance(bfn.body[0].value, _ast.Constant)
                and isinstance(bfn.body[0].value.value, str)) else 0
    bfn.body[ins:ins] = [call]
    new_code = _ast.unparse(vtree) + "\n\n" + _ast.unparse(btree)
    try:
        _ast.parse(new_code)
    except SyntaxError:
        return body_code, -1
    return new_code, n_asserts


def clause_lines_B(tid, nodes):
    t = nodes[tid]
    return [c["text"] for c in t["contract_nodes"] if c.get("is_input_precondition")]


def clause_lines_C(tid, specs):
    s = specs.get(tid) or {"specs": [], "unparsed": []}
    lines = [json.dumps({k: v for k, v in sp.items() if k != "clause"}, ensure_ascii=False)
             for sp in s["specs"]]
    lines += [f'{{"kind": "unparsed", "clause": "{u}"}}' for u in s["unparsed"]]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--conditions", default="nl", choices=["nl", "json"],
                   help="how conditions are presented: nl (method) | json (rule-parser spec)")
    p.add_argument("--limit", type=int, default=340)
    p.add_argument("--tag", default=DEFAULT_SETTING, help="setting name; names the output directory")
    p.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--resume", default="True")
    p.add_argument("--selected_path", default="", help="selected bodies; default the setting's selected.json")
    p.add_argument("--clauses_path", default="", help="condition list from stage (i); "
                   "default the setting's conditions.json, or artifacts/condition_nodes.json "
                   "with --from_reference")
    p.add_argument("--from_reference", action="store_true",
                   help="comparison arm: take the conditions from the reference contract text")
    p.add_argument("--vform", default="asserts", choices=["asserts", "func"],
                   help="assert lines per condition (method) | a separate validator function (ablation)")
    p.add_argument("--with_desc", default="False",
                   help="ablation: also show the problem description, as reference only")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (method)")
    p.add_argument("--no_assume", default="False",
                   help="do not enforce 'you may assume ...' clauses; they state an assumption, "
                        "not a rejection requirement (method: True)")
    p.add_argument("--cov_retry", default="False",
                   help="ablation: if fewer asserts survive than conditions, regenerate once and merge")
    p.add_argument("--quotes_path", default="",
                   help="ablation: {tid: [[condition, quote], ...]} to attach a source quote per condition")
    args = p.parse_args()
    root = Path(".")
    resume = str(args.resume).lower() == "true"
    with_desc = str(args.with_desc).lower() == "true"
    system = SYSTEM
    if str(args.no_assume).lower() == "true":
        system = SYSTEM + ("\n- If a clause merely states an assumption you are allowed to make "
                           "(e.g. 'you may assume ...'), do NOT write an assert for it — "
                           "assumptions are not rejection requirements.")

    nodes = ({t["id"]: t for t in json.load(open("artifacts/condition_nodes.json"))}
             if args.from_reference else {})   # reference contract: comparison arm only
    descs = ({str(t["id"]): (t.get("description") or "")
              for t in json.load(open("dataset/contracteval/test.json"))} if with_desc else {})
    specs = json.load(open("artifacts/condition_specs.json")) if args.conditions == "json" else {}
    cond_path = args.clauses_path or ("" if args.from_reference
                                      else run_path(args.tag, Path(args.model_name).name, "conditions"))
    self_clauses = json.load(open(cond_path)) if cond_path else None
    quotes = json.load(open(args.quotes_path)) if args.quotes_path else None
    mshort = Path(args.model_name).name
    sel_path = args.selected_path or run_path(args.tag, mshort, "selected")
    sel = json.load(open(sel_path))[: args.limit]

    out_path = root / run_path(args.tag, mshort, "assertions_raw")
    out_dir = out_path.parent
    existing = {}
    if resume and out_path.exists():
        try:
            existing = {str(r["id"]): r for r in json.load(open(out_path))}
        except Exception:
            existing = {}

    todo = [r for r in sel if not (resume and str(r["id"]) in existing)]
    print(f"[assert] {len(sel)} tasks ({len(sel)-len(todo)} reused)")
    if todo:
        tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir,
                                            trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, cache_dir=args.cache_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

    raw_path = out_dir / "assertions_log.json"
    raw_log = json.load(open(raw_path)) if raw_path.exists() else []
    results = [existing[str(r["id"])] for r in sel if str(r["id"]) in existing]
    for r in tqdm(todo, desc="assert", unit="task"):
        tid = str(r["id"])
        if self_clauses is not None:
            clauses = self_clauses.get(tid) or []
        else:
            clauses = (clause_lines_B(tid, nodes) if args.conditions == "nl"
                       else clause_lines_C(tid, specs))
        rec = dict(r)
        if not clauses:
            rec.update(n_asserts=0, raw_assertions="")
            results.append(rec)
            json.dump(results, open(out_path, "w"), indent=1)
            continue
        qmap = quotes.get(tid) if quotes else None
        if qmap:                                    # one source quote per condition (ablation)
            lines_n = []
            for i, c in enumerate(clauses):
                q = next((pq for pc, pq in qmap if pc == str(c) and pq), None)
                lines_n.append(f"{i+1}. {c}" + (f'\n   source: "{q}"' if q else ""))
            numbered = "\n".join(lines_n)
        else:
            numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(clauses))
        desc_block = ""
        if with_desc and descs.get(tid):
            desc_block = ("Problem description (REFERENCE ONLY — use it to interpret the "
                          "clauses; do NOT add any check that is not in the clauses):\n"
                          f"{descs[tid]}\n\n")
        if args.vform == "func":
            user = (f"Function:\n```python\n{r['code']}\n```\n\n"
                    f"Contract clauses:\n{numbered}\n\n"
                    f"Write the validator function.")
            prompt = build_chat_prompt(tok, SYSTEM_FUNC, user, True)
        else:
            user = (f"Function:\n```python\n{r['code']}\n```\n\n"
                    f"{desc_block}"
                    f"Contract clauses:\n{numbered}\n\n"
                    f"Write one assert per clause.")
            prompt = build_chat_prompt(tok, system, user, True)
        enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
        do_sample = args.temperature > 0
        with torch.no_grad():
            out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                                 temperature=args.temperature if do_sample else None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        if args.vform == "func":
            new_code, n = compose_validator_func(r["code"], r.get("entry_point") or "", text)
        else:
            rec["pass1_raw"] = text
            lines = [l.strip().lstrip("0123456789. ") for l in text.split("\n")]
            asserts = [l for l in lines if l.startswith("assert")]
            new_code, n = insert_asserts(r["code"], r.get("entry_point") or "", asserts)
            rec["pass1_code"], rec["pass1_asserts"], rec["cov_retry_fired"] = new_code, n, False
            # coverage retry: fewer surviving asserts than conditions -> ask once more
            if str(args.cov_retry).lower() == "true" and 0 <= n < len(clauses):
                user2 = (user + f"\n\nYour previous answer:\n{text}\n\n"
                         f"Only {max(n,0)} of {len(clauses)} clauses got a surviving assert. "
                         "Write the FULL list again, one assert per clause, using the actual "
                         "parameter names of the function.")
                prompt2 = build_chat_prompt(tok, system, user2, True)
                enc2 = tok(prompt2, return_tensors="pt").to(next(model.parameters()).device)
                with torch.no_grad():
                    out2 = model.generate(enc2.input_ids, attention_mask=enc2.attention_mask,
                                          max_new_tokens=args.max_new_tokens, do_sample=False,
                                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
                text2 = tok.decode(out2[0, enc2.input_ids.shape[1]:], skip_special_tokens=True)
                lines2 = [l.strip().lstrip("0123456789. ") for l in text2.split("\n")]
                asserts2 = [l for l in lines2 if l.startswith("assert")]
                new_code2, n2 = insert_asserts(r["code"], r.get("entry_point") or "",
                                               asserts + asserts2)
                rec["cov_retry_fired"] = True
                rec["pass2_raw"] = text2
                if n2 > n:
                    new_code, n, text = new_code2, n2, text + "\n--retry--\n" + text2
        rec.update(code=new_code, n_asserts=n, raw_assertions=text[:500])
        results.append(rec)
        json.dump(results, open(out_path, "w"), indent=1)
        raw_log.append({"id": tid, "pass1_raw": rec.get("pass1_raw", ""),
                        "pass2_raw": rec.get("pass2_raw"), "clauses": clauses})
        json.dump(raw_log, open(raw_path, "w"), indent=1)
    n_att = sum(1 for r in results if (r.get("n_asserts") or 0) > 0)
    print(f"[assert] done {len(results)} | asserts inserted in {n_att} tasks -> {out_path}")


if __name__ == "__main__":
    main()
