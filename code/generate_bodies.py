"""Stage (ii): generate functional body candidates from the functional view.

The instruction is the plain functional one (no contract, no assert wording) and the problem
description is the masked view produced by stage (i), so the candidates are never conditioned on
contract text. The original contract-free `prompt_base` is never read; the view is derived only
from `prompt_cs` + the extracted conditions.

    --n_samples 1                  greedy candidate
    --n_samples 4 --temperature .7 four sampled candidates (`samples` field)

Writes output/<setting>/<model>/bodies_<out_tag>.json
"""
import argparse
import os
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_baseline import generate_code, _parse_ok
from utils.instruction.baseline import _INSTRUCTION, _OUTPUT_INSTRUCTION
from utils.contract_eval import extract_python_code
from utils.dataset import load_contract_dataset
from utils.gen_common import build_chat_prompt
from utils.paths import run_path, DEFAULT_SETTING


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--project_root", default=".")
    p.add_argument("--tag", default=DEFAULT_SETTING, help="setting name; names the output directory")
    p.add_argument("--masked_file", default="", help="functional view; default the setting's view.json")
    p.add_argument("--ids_file", default="dataset/contracteval/eval_tasks_340.json")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--no_think", default="True")
    p.add_argument("--resume", default="True")
    p.add_argument("--n_samples", type=int, default=1,
                   help="if >1, store N temperature samples in the `samples` field")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out_tag", default="greedy", choices=["greedy", "sampled"])
    args = p.parse_args()
    root = Path(args.project_root)
    mshort = Path(args.model_name).name
    no_think = str(args.no_think).lower() == "true"
    resume = str(args.resume).lower() == "true"

    view_file = args.masked_file or run_path(args.tag, mshort, "view")
    masked = json.load(open(root / view_file))
    spec = json.load(open(root / args.ids_file))
    keep = list(map(str, spec["ids"] if isinstance(spec, dict) else spec))[: args.limit]
    tasks = {str(t["id"]): t for t in load_contract_dataset(root / "dataset/contracteval/test.json")}
    todo_ids = [i for i in keep if i in tasks and i in masked]
    print(f"[bodies] {len(todo_ids)} tasks (frozen-340 head {args.limit})")

    out_path = root / run_path(args.tag, mshort, f"bodies_{args.out_tag}")
    existing = {}
    if resume and out_path.exists():
        try:
            existing = {str(r["id"]): r for r in json.load(open(out_path))}
        except Exception:
            existing = {}
    def _reusable(r):
        if r is None or r.get("code") is None:
            return False
        return len(r.get("samples") or [r["code"]]) >= args.n_samples

    if all(_reusable(existing.get(i)) for i in todo_ids):
        print(f"[bodies] all present -> skip. {out_path}")
        return

    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir,
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    system = f"{_INSTRUCTION['base']}\n\n{_OUTPUT_INSTRUCTION}"
    results = []
    for tid in tqdm(todo_ids, desc="bodies", unit="task"):
        prev = existing.get(tid)
        if resume and _reusable(prev):
            results.append(prev); continue
        t = tasks[tid]
        desc = masked[tid]["masked"].strip()
        user = f"Method Name: {t.get('entry_point') or tid}\nProblem Description:\n{desc}"
        prompt = build_chat_prompt(tok, system, user, no_think)
        if args.n_samples <= 1:                                  # greedy
            code, n_tok = generate_code(model, tok, prompt, args.max_new_tokens, False, 0.0)
            rec = {"id": tid, "entry_point": t.get("entry_point"), "code": code,
                   "gen_tokens": n_tok, "condition": "functional view"}
        else:                                                    # temperature samples
            samples, toks = [], 0
            for _ in range(args.n_samples):
                c, n_tok = generate_code(model, tok, prompt, args.max_new_tokens,
                                         True, args.temperature)
                samples.append(c); toks += n_tok
            rec = {"id": tid, "entry_point": t.get("entry_point"),
                   "code": samples[0], "samples": samples, "gen_tokens": toks,
                   "condition": f"functional view n={args.n_samples} T={args.temperature}"}
        ok, why = _parse_ok(rec["code"], t.get("entry_point") or "")
        rec.update(parsed=ok, parse_error=why)
        results.append(rec)
        json.dump(results, open(out_path, "w"), indent=1)     # crash-safe incremental
    json.dump(results, open(out_path, "w"), indent=1)
    n_ok = sum(r["parsed"] for r in results)
    print(f"[bodies] done {len(results)} tasks, parsed {n_ok}. -> {out_path}")


if __name__ == "__main__":
    main()
