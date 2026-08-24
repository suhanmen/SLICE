"""Stage (i): ask the model which input preconditions the problem description states.

The only input is `prompt_cs` (plus the signature for the `refined` prompt); the reference
contract text is never read, so the conditions are self-extracted. Greedy decoding.

    python code/identify_contract_conditions.py --model_name <hf_id> [--variant refined]

Writes output/<setting>/<model>/conditions.json  as {task id: [condition, ...]}
"""
import argparse, json, os, re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.gen_common import build_chat_prompt
from utils.paths import run_path, DEFAULT_SETTING

# Prompt variants, in order of decreasing recall pressure. `refined` is the method; the others
# are the precision/recall ablation. Pushing for completeness ("list every one") raises the
# condition count but also over-rejection, so `refined` keeps the plain instruction and adds only
# narrow prohibitions against the over-statement patterns observed in its output.
SYSTEM_RECALL = (
    "You extract input-contract clauses from a programming problem description.\n"
    "List ONLY the preconditions that the INPUTS must satisfy, as stated or clearly implied "
    "by the description.\n"
    "COMPLETENESS RULES (most important — missing a condition is the worst failure):\n"
    "- Do NOT summarize or merge conditions. Output EVERY distinct precondition as its own "
    "separate line, even if there are many.\n"
    "- Systematically check ALL of these categories and include every one that appears:\n"
    "    (1) type of EACH argument, (2) numeric ranges with their exact bounds, "
    "(3) length / non-emptiness conditions, (4) allowed characters or values, "
    "(5) per-element conditions of lists/tuples/dicts.\n"
    "- If a condition applies to multiple arguments, write one line PER argument.\n"
    "OTHER RULES:\n"
    "- One clause per line, plain English, e.g. 'x must be an integer.'\n"
    "- Do NOT invent constraints that are not in the description.\n"
    "- Do NOT include behavioral requirements (what the function returns/does).\n"
    "- If there are no input preconditions, output exactly: NONE"
)

SYSTEM_EXPLICIT = (
    "You extract input-contract clauses from a programming problem description.\n"
    "List ONLY the preconditions on the INPUTS that the description EXPLICITLY states.\n"
    "PRECISION RULES (most important — a wrong or invented condition is the worst failure):\n"
    "- Include a condition ONLY if the description states it in so many words.\n"
    "- Do NOT infer, generalize, or expand conditions. Do NOT add 'obvious' constraints "
    "the description never mentions.\n"
    "- If you are unsure whether something is required, OMIT it.\n"
    "OTHER RULES:\n"
    "- One clause per line, plain English, e.g. 'x must be an integer.'\n"
    "- Do NOT include behavioral requirements (what the function returns/does).\n"
    "- If there are no input preconditions, output exactly: NONE"
)

SYSTEM_PRECBAL = (
    "You extract input-contract clauses from a programming problem description.\n"
    "List ONLY the preconditions that the INPUTS must satisfy, as stated by the description.\n"
    "Check these categories, but include a condition ONLY when the description actually "
    "states it: (1) argument types, (2) numeric ranges, (3) length / non-emptiness, "
    "(4) allowed characters or values, (5) per-element conditions of containers.\n"
    "PRECISION RULES:\n"
    "- Do NOT invent or infer constraints the description does not state. When unsure, omit.\n"
    "- Do NOT add stricter bounds than stated.\n"
    "OTHER RULES:\n"
    "- One clause per line, plain English, e.g. 'x must be an integer.'\n"
    "- Do NOT include behavioral requirements (what the function returns/does).\n"
    "- If there are no input preconditions, output exactly: NONE"
)

SYSTEM_REFINED = (
    "You extract input-contract clauses from a programming problem description.\n"
    "List ONLY the preconditions that the INPUTS must satisfy, as stated by the description.\n"
    "Rules:\n"
    "- One clause per line, plain English, e.g. 'x must be an integer.'\n"
    "- Do NOT invent constraints that are not in the description.\n"
    "- Do NOT include behavioral requirements (what the function returns/does).\n"
    "- Refer to arguments by their EXACT names from the function signature.\n"
    "DO NOT OVER-STATE (most common mistakes — avoid them):\n"
    "- Do NOT add non-emptiness. 'must be a string' means '' is allowed.\n"
    "    BAD:  'text must be a non-empty string'   GOOD: 'text must be a string'\n"
    "- Keep element-level conditions element-level; do NOT promote them to a container-type "
    "requirement.\n"
    "    Description: 'all elements in the list must be integers'\n"
    "    BAD:  'nums must be a list of integers'   GOOD: 'all elements of nums must be integers'\n"
    "- Do NOT add relations BETWEEN arguments (e.g. 'n must not exceed m', 'start <= end') "
    "unless the description states the relation.\n"
    "- Do NOT merge per-word/per-element conditions into a whole-value condition.\n"
    "    Description: 'each word consists of letters'\n"
    "    BAD:  'sentence must contain only letters' (kills spaces)\n"
    "    GOOD: 'each word in sentence must consist only of letters'\n"
    "- Side notes about formats/examples are not preconditions — skip them.\n"
    "- If there are no input preconditions, output exactly: NONE"
)


# The plain instruction `refined` is built on.
SYSTEM_RECALLV1 = (
    "You extract input-contract clauses from a programming problem description.\n"
    "List ONLY the preconditions that the INPUTS must satisfy (types, ranges, lengths, "
    "allowed characters/values), as stated or clearly implied by the description.\n"
    "Rules:\n"
    "- One clause per line, plain English, e.g. 'x must be an integer.'\n"
    "- Do NOT invent constraints that are not in the description.\n"
    "- Do NOT include behavioral requirements (what the function returns/does).\n"
    "- If there are no input preconditions, output exactly: NONE"
)

VARIANTS = {"recall": (SYSTEM_RECALL, 12), "recallv1": (SYSTEM_RECALLV1, 8),
            "explicit": (SYSTEM_EXPLICIT, 8),
            "precbal": (SYSTEM_PRECBAL, 10), "refined": (SYSTEM_REFINED, 12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    ap.add_argument("--variant", default="refined", choices=list(VARIANTS))
    ap.add_argument("--tag", default=DEFAULT_SETTING, help="setting name; names the output directory")
    args = ap.parse_args()
    mshort = Path(args.model_name).name
    system, cap = VARIANTS[args.variant]

    frozen = json.load(open("dataset/contracteval/eval_tasks_340.json"))["ids"]
    d = {str(r["id"]): r for r in json.load(open("dataset/contracteval/test.json"))}

    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir,
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    out_path = Path(run_path(args.tag, mshort, "conditions"))
    out = {}
    if out_path.exists():
        try: out = json.load(open(out_path))
        except Exception: out = {}
    todo = [t for t in frozen if t not in out]
    for tid in tqdm(todo, unit="task"):
        pcs = (d[tid].get("prompt_cs") or "").strip()
        # The signature is interface information, not contract text. Mbpp descriptions carry no
        # def line, so without it the model invents argument names.
        sig = (d[tid].get("signature") or "").strip()
        sig_line = f"Function signature: {sig}\n" if (args.variant == "refined" and sig) else ""
        user = f"{sig_line}Problem description:\n{pcs}\n\nList the input preconditions."
        prompt = build_chat_prompt(tok, system, user, True)
        enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            o = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                               max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        lines = [re.sub(r"^[\-\*\d.\)\s]+", "", l).strip() for l in text.split("\n")]
        clauses = []
        for l in lines:
            if not l or l.upper() == "NONE":
                continue
            c = l.split("||", 1)[0].strip()
            if len(c) > 8:
                clauses.append(c)
        clauses = clauses[:cap]                   # per-variant cap
        out[tid] = clauses
        json.dump(out, open(out_path, "w"), indent=1)
    print(f"[extract] {len(out)} tasks -> {out_path}")


if __name__ == "__main__":
    main()
