"""Prompt-level baselines for API models — the twin of run_baseline.py.

Builds the identical prompt (utils.instruction.baseline.build_baseline_messages) and writes the
identical output schema, so an API model drops straight into the comparison table; the only
difference is that generation is a chat-completion call instead of a local forward pass.

Reasoning and *-nano models reject `max_tokens` (use `max_completion_tokens`) and reject any
temperature other than the default, so temperature is omitted rather than set.

    python code/run_baseline_api.py --model_name <model> --condition cs

Writes output/base_<condition>/<model>/final.json
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from utils.instruction.baseline import build_baseline_messages, CONDITIONS
from utils.contract_eval import contract_evaluate, extract_python_code
from utils.paths import run_dir
from utils.dataset import load_contract_dataset


def _load_key(args) -> str:
    if args.api_key:
        return args.api_key
    if args.api_key_file and Path(args.api_key_file).exists():
        for line in Path(args.api_key_file).read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line and any(k in line for k in ("OPENAI", "OPEN_API")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_API_KEY") or ""


def _call(client, model, system, user, max_completion_tokens, temperature,
          reasoning_effort, retries=5):
    """One chat completion with retry/backoff. Returns raw assistant text ('' on failure)."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_completion_tokens=max_completion_tokens,   # reasoning models reject max_tokens
    )
    if temperature is not None:                        # nano defaults temp=1 and rejects others
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:                         # rate limit / transient / server
            if attempt == retries - 1:
                return f"__API_ERROR__: {e}"
            time.sleep(2 ** attempt)
    return ""


def _parse_ok(code: str, entry_point: str):
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


def main():
    p = argparse.ArgumentParser(description="OpenAI API baseline on ContractEval, scored by contract_eval")
    p.add_argument("--model", required=True, help="e.g. gpt-5.4-nano")
    p.add_argument("--dataset", default="contracteval")
    p.add_argument("--condition", default="cs", choices=list(CONDITIONS))
    p.add_argument("--project_root", default=".")
    p.add_argument("--max_completion_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=None,
                   help="omit for nano/reasoning models (they only allow the default)")
    p.add_argument("--reasoning_effort", default="", help="minimal|low|medium|high (if supported)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0 = full dataset (364)")
    p.add_argument("--resume", default="True")
    p.add_argument("--eval", default="False", help="inline contract_eval after generation")
    p.add_argument("--api_key", default="")
    p.add_argument("--api_key_file", default="",
                   help="file holding OPENAI_API_KEY=... (an `export` prefix is accepted)")
    args = p.parse_args()

    resume = str(args.resume).lower() == "true"
    root = Path(args.project_root)
    tasks = load_contract_dataset(root / "dataset" / args.dataset / "test.json")
    if args.limit:
        tasks = tasks[: args.limit]

    model_base = args.model.replace("/", "--")
    out_dir = Path(run_dir(f"base_{args.condition}", model_base))
    out_path = out_dir / "final.json"

    existing = {}
    if resume and out_path.exists():
        try:
            for r in json.load(open(out_path, encoding="utf-8")):
                existing[str(r.get("id"))] = r
        except Exception:
            existing = {}

    def _reusable(r):
        return r is not None and r.get("code") is not None and not str(r.get("raw", "")).startswith("__API_ERROR__")

    todo = [t for t in tasks if not (resume and _reusable(existing.get(str(t.get("id")))))]
    print(f"[api:{args.model}:{args.condition}] {len(tasks)-len(todo)} reused, {len(todo)} to generate "
          f"(concurrency={args.concurrency}) -> {out_path}")
    if not todo:
        print("[api] nothing to do.");
        if str(args.eval).lower() != "true":
            return

    key = _load_key(args)
    if todo and not key:
        raise SystemExit("No API key. Set OPENAI_API_KEY, or pass --api_key / --api_key_file.")

    results_by_id = dict(existing)
    if todo:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        def work(t):
            system, user = build_baseline_messages(t, args.condition)
            raw = _call(client, args.model, system, user, args.max_completion_tokens,
                        args.temperature, args.reasoning_effort or None)
            code = extract_python_code(raw)
            ok, _ = _parse_ok(code or "", t.get("entry_point") or "")
            return str(t.get("id")), {"id": t.get("id"), "code": code, "samples": [code],
                                      "n_validator_tokens": 0, "parsed": ok, "raw": raw}

        done = err = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, t): t for t in todo}
            bar = tqdm(as_completed(futs), total=len(todo),
                       desc=f"api:{args.model}:{args.condition}", unit="task")
            for fut in bar:
                tid, rec = fut.result()
                results_by_id[tid] = rec
                done += 1
                if str(rec.get("raw", "")).startswith("__API_ERROR__"):
                    err += 1
                    bar.write(f"  [API err] {tid}: {rec['raw'][:120]}")
                bar.set_postfix(errors=err)
                if done % 25 == 0:
                    json.dump(list(results_by_id.values()), open(out_path, "w"),
                              indent=2, ensure_ascii=False)   # incremental save
        print(f"[api] generated {done} (errors={err})")

    # keep dataset order
    results = [results_by_id[str(t.get("id"))] for t in tasks if str(t.get("id")) in results_by_id]
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    n_parsed = sum(int(bool(r.get("parsed"))) for r in results)
    print(f"[api] wrote {len(results)} -> {out_path} | parsed {n_parsed}/{len(results)}")

    if str(args.eval).lower() == "true":
        report = contract_evaluate(results, tasks, dataset=args.dataset,
                                   project_root=str(root), ks=(1,))
        agg = report["aggregate"]
        print("\n[api] evaluation (strict / func / CSR / EASY / HARD / over_rej):")
        for k in ("strict_pass@1", "func_total@1", "csr@1", "csr_type@1",
                  "csr_value@1", "over_rejection_total@1"):
            print(f"  {k:<24} {agg.get(k)}")
        json.dump(report, open(out_dir / f"{model_base}_eval.json", "w"),
                  indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
