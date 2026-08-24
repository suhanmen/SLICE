"""Prompt-level baselines — ContractEval's own conditions (base / cs / eas).

A single pass with no pipeline: build the condition prompt, generate the whole function,
extract the code block, and score it with the same utils.contract_eval the method uses, so
both sides share one metric. `eas` reproduces ContractEval's "Contract Prompt", which puts
the contract-violating inputs in the prompt.

    PYTHONPATH=code python code/run_baseline.py --model_name <hf_id> --condition cs

Writes output/base_<condition>/<model>/final.json
"""

import argparse
import os
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.instruction.baseline import build_baseline_messages, CONDITIONS
from utils.contract_eval import contract_evaluate, extract_python_code
from utils.paths import run_dir
from utils.dataset import load_contract_dataset
from utils.gen_common import build_chat_prompt


def _parse_ok(code: str, entry_point: str):
    """Did the generated text parse into a runnable function? Returns (ok, reason)."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg}"
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return False, "no function def"
    if entry_point and entry_point not in funcs:
        return False, f"entry_point '{entry_point}' not defined (got {funcs})"
    return True, ""


@torch.no_grad()
def generate_code(model, tokenizer, prompt, max_new_tokens, do_sample, temperature):
    """ContractEval-style: chat prompt -> model generates the COMPLETE function ->
    extract the ```python``` code block (strip any <think> first)."""
    import re, os
    # In think-prefill mode, prime the opening code fence too: without it these models emit
    # prose and code with no fence. The fence is restored after decoding, for the extractor.
    _pf = os.environ.get("THINK_PREFILL") == "1"
    if _pf and not prompt.rstrip().endswith("```python"):
        prompt = prompt + "```python\n"
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    ids = enc.input_ids
    kwargs = dict(attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(ids, **kwargs)[0, ids.shape[1]:]
    text = tokenizer.decode(out, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).split("<think>")[0]
    if _pf:
        text = "```python\n" + text
    return extract_python_code(text), len(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", default="contracteval")
    p.add_argument("--condition", required=True, choices=list(CONDITIONS))
    p.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--project_root", default=".")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--do_sample", default="False")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--n_samples", type=int, default=1, help=">1 enables sampling for pass@k")
    p.add_argument("--no_think", default="True", help="disable <think> for reasoning models")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids_file", default="", help="JSON list, or {test:[...]} — restrict to these task ids")
    p.add_argument("--adapter", default="", help="LoRA adapter dir to load on top of the base model")
    p.add_argument("--out_tag", default="", help="suffix on the output filename (e.g. sft) to avoid clobber")
    p.add_argument("--eval", default="False", help="inline eval; default off — use eval_main.sh")
    p.add_argument("--resume", default="True",
                   help="reuse existing per-task results; only generate missing task_ids "
                        "(all present -> skip without loading the model). 'False' = regenerate all")
    args = p.parse_args()

    do_sample = str(args.do_sample).lower() == "true"
    no_think = str(args.no_think).lower() == "true"
    resume = str(args.resume).lower() == "true"
    n_samples = max(1, args.n_samples)
    sample_do_sample = do_sample or n_samples > 1   # need sampling for pass@k (n>1)
    print(f"[baseline] model={args.model_name} condition={args.condition} dataset={args.dataset}")

    path = Path(args.project_root) / "dataset" / args.dataset / "test.json"
    tasks = load_contract_dataset(path)
    if args.ids_file:                     # restrict to a held-out id set (e.g. the SFT test 100)
        spec = json.load(open(args.ids_file))
        keep = set(map(str, spec.get("test", spec) if isinstance(spec, dict) else spec))
        tasks = [t for t in tasks if str(t.get("id")) in keep]
        print(f"[baseline] ids_file -> {len(tasks)} tasks")
    if args.limit:
        tasks = tasks[: args.limit]

    # New layout: output/baseline/<dataset>/<condition>/<model>[__tag]_output.json
    # (dataset first -> no duplicate <dataset> dir under every condition).
    out_dir = Path(run_dir(f"base_{args.condition}", model_base))
    model_base = Path(args.model_name).name + (f"__{args.out_tag}" if args.out_tag else "")
    out_path = out_dir / "final.json"

    # --- resume: load existing per-task results; reuse any task_id already covered ---
    existing = {}
    if resume and out_path.exists():
        try:
            for r in json.load(open(out_path, encoding="utf-8")):
                existing[str(r.get("id"))] = r
        except Exception:
            existing = {}

    def _reusable(r):  # present AND has enough samples for the requested pass@k
        return (r is not None and r.get("code") is not None
                and len(r.get("samples") or [r.get("code")]) >= n_samples)

    todo = [t for t in tasks if not (resume and _reusable(existing.get(str(t.get("id")))))]
    if not todo:
        print(f"[baseline:{args.condition}] all {len(tasks)} task_ids already present "
              f"-> skip (model not loaded). {out_path}")
        return

    print(f"[baseline:{args.condition}] {len(tasks)-len(todo)} reused, {len(todo)} to generate")

    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:                      # load LoRA adapter on top of the base model
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[baseline] loaded LoRA adapter: {args.adapter}")
    model.eval()

    results = []
    n_parsed = 0
    bar = tqdm(tasks, desc=f"baseline:{args.condition}", unit="task")
    for t in bar:
        prev = existing.get(str(t.get("id")))
        if resume and _reusable(prev):      # keep already-generated task verbatim
            results.append(prev)
            n_parsed += int(bool(prev.get("parsed")))
            continue
        system, user = build_baseline_messages(t, args.condition)
        prompt = build_chat_prompt(tok, system, user, no_think)
        samples, first_tok = [], 0
        for s in range(n_samples):
            code, n_tok = generate_code(model, tok, prompt,
                                        args.max_new_tokens, sample_do_sample, args.temperature)
            samples.append(code)
            if s == 0:
                first_tok = n_tok
        ok, reason = _parse_ok(samples[0], t.get("entry_point") or "")
        n_parsed += int(ok)
        bar.set_postfix(parsed=f"{n_parsed}/{len(results)+1}")
        if not ok:  # only surface parse failures; OK ones stay quiet (just the tqdm bar)
            tqdm.write(f"  [parse FAIL] {t.get('id')} (n={n_samples}, tok={first_tok})  <- {reason}")
        results.append({"id": t.get("id"), "code": samples[0], "samples": samples,
                        "n_validator_tokens": first_tok, "parsed": ok})
        # Save every task, so a crash mid-run leaves the finished ones for --resume.
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    rate = n_parsed / len(results) if results else 0.0
    print(f"\n[baseline:{args.condition}] parsed {n_parsed}/{len(results)} "
          f"({rate:.1%}) into a runnable function")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if str(args.eval).lower() == "true":
        report = contract_evaluate(results, tasks)
        report["aggregate"]["parsed"] = f"{n_parsed}/{len(results)}"
        report["aggregate"]["parse_rate"] = rate
        print(f"\n[baseline:{args.condition}] evaluation:")
        print(json.dumps(report["aggregate"], indent=2))
        with open(out_dir / f"{model_base}_eval.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
