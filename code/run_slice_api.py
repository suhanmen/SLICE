"""SLICE for API models: the three LLM stages, run against an OpenAI-compatible endpoint.

    conditions : stage (i)   self-extract the contract conditions
    bodies     : stage (ii)  candidate bodies from the functional view
    assertions : stage (iii) one assert per condition, body-aware

Everything else (the specification graph, selection, renaming, screening, scoring) is CPU-side and
shared with the local chain, so those stages are run from the same modules.

Paths match the local chain exactly, which is what lets the shared stages be reused: every
file goes to output/<setting>/<model>/ (see utils/paths.py).

    python code/run_slice_api.py --model <model> --stage conditions|bodies|assertions [--tag SLICE]

Note on temperature: these models fix it, so a greedy/T0.7 split is not available. The candidate
pool is instead one anchor call plus four further calls at the model's own default, which is why
the API arm's candidates are less diverse than the local one.
"""
import argparse, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from utils.dataset import load_contract_dataset
from utils.contract_eval import extract_python_code
from run_baseline_api import _load_key, _call, _parse_ok
from identify_contract_conditions import VARIANTS
from utils.instruction.baseline import _INSTRUCTION, _OUTPUT_INSTRUCTION
import generate_assertions as _asrt
from utils.paths import run_path, DEFAULT_SETTING

CAP = 5                                     # candidates per task: 1 anchor + 4 samples

FROZEN = None


def frozen():
    global FROZEN
    if FROZEN is None:
        FROZEN = json.load(open("dataset/contracteval/eval_tasks_340.json"))["ids"]
    return FROZEN


def stage_conditions(args, client, tasks):
    system, cap = VARIANTS[args.variant]
    out_path = Path(run_path(args.tag, args.model, "conditions"))
    out = json.load(open(out_path)) if out_path.exists() else {}
    todo = [t for t in frozen() if t not in out]
    d = {str(r["id"]): r for r in tasks}

    def gen(tid):
        pcs = (d[tid].get("prompt_cs") or "").strip()
        sig = (d[tid].get("signature") or "").strip()
        sig_line = f"Function signature: {sig}\n" if (args.variant == "refined" and sig) else ""
        user = f"{sig_line}Problem description:\n{pcs}\n\nList the input preconditions."
        text = _call(client, args.model, system, user, 400, None, "")
        lines = [re.sub(r"^[\-\*\d.\)\s]+", "", l).strip() for l in (text or "").split("\n")]
        clauses = [l.split("||", 1)[0].strip() for l in lines
                   if l and l.upper() != "NONE" and len(l) > 8]
        return tid, clauses[:cap]

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(gen, t) for t in todo]
        for f in tqdm(as_completed(futs), total=len(futs), unit="task"):
            tid, cl = f.result(); out[tid] = cl
            if len(out) % 20 == 0:
                json.dump(out, open(out_path, "w"), indent=1)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[conditions] {len(out)} tasks, {sum(len(v) for v in out.values())} conditions "
          f"-> {out_path}")


def stage_bodies(args, client, tasks):
    view = json.load(open(run_path(args.tag, args.model, "view")))
    d = {str(r["id"]): r for r in tasks}
    system = f"{_INSTRUCTION['base']}\n\n{_OUTPUT_INSTRUCTION}"
    g_path = Path(run_path(args.tag, args.model, "bodies_greedy"))
    n_path = Path(run_path(args.tag, args.model, "bodies_sampled"))
    g_out = {str(r["id"]): r for r in json.load(open(g_path))} if g_path.exists() else {}
    n_out = {str(r["id"]): r for r in json.load(open(n_path))} if n_path.exists() else {}
    todo = [t for t in frozen()
            if not (t in g_out and len((n_out.get(t) or {}).get("samples") or []) >= CAP - 1)]

    def gen(tid):
        t = d[tid]
        desc = (view.get(tid) or {}).get("masked", t.get("prompt_cs") or "").strip()
        user = f"Method Name: {t.get('entry_point') or tid}\nProblem Description:\n{desc}"
        outs = []
        for _ in range(CAP):
            raw = _call(client, args.model, system, user, 2048, None, "")
            outs.append(extract_python_code(raw or "") or "")
        return tid, outs

    def save():
        json.dump(list(g_out.values()), open(g_path, "w"), indent=1)
        json.dump(list(n_out.values()), open(n_path, "w"), indent=1)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(gen, t) for t in todo]
        for f in tqdm(as_completed(futs), total=len(futs), unit="task"):
            tid, outs = f.result()
            t = d[tid]
            ok, why = _parse_ok(outs[0], t.get("entry_point") or "")
            g_out[tid] = {"id": tid, "entry_point": t.get("entry_point"), "code": outs[0],
                          "gen_tokens": 0, "condition": "functional view (api)",
                          "parsed": ok, "parse_error": why}
            n_out[tid] = {"id": tid, "entry_point": t.get("entry_point"), "code": outs[1],
                          "samples": outs[1:], "gen_tokens": 0,
                          "condition": f"functional view (api) n={CAP-1}",
                          "parsed": True, "parse_error": ""}
            if len(g_out) % 5 == 0:
                save()
    save()
    print(f"[bodies] {len(g_out)} tasks -> {g_path}, {n_path}")


def stage_assertions(args, client, tasks):
    conditions = json.load(open(run_path(args.tag, args.model, "conditions")))
    sel = json.load(open(run_path(args.tag, args.model, "selected")))
    system = _asrt.SYSTEM + ("\n- If a clause merely states an assumption you are allowed to make "
                             "(e.g. 'you may assume ...'), do NOT write an assert for it — "
                             "assumptions are not rejection requirements.")
    out_path = Path(run_path(args.tag, args.model, "assertions_raw"))
    done = {str(r["id"]): r for r in json.load(open(out_path))} if out_path.exists() else {}
    todo = [r for r in sel if str(r["id"]) not in done]

    def gen(r):
        tid = str(r["id"])
        cl = conditions.get(tid) or []
        rec = dict(r)
        if not cl:
            rec.update(n_asserts=0, raw_assertions="")
            return tid, rec
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cl))
        user = (f"Function:\n```python\n{r['code']}\n```\n\n"
                f"Contract clauses:\n{numbered}\n\n"
                f"Write one assert per clause.")
        text = _call(client, args.model, system, user, 512, None, "") or ""
        lines = [l.strip().lstrip("0123456789. ") for l in text.split("\n")]
        asserts = [l for l in lines if l.startswith("assert")]
        code, n = _asrt.insert_asserts(r["code"], r.get("entry_point") or "", asserts)
        rec.update(code=code, n_asserts=n, raw_assertions=text[:500])
        return tid, rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(gen, r) for r in todo]
        for f in tqdm(as_completed(futs), total=len(futs), unit="task"):
            tid, rec = f.result(); done[tid] = rec
            if len(done) % 20 == 0:
                json.dump(list(done.values()), open(out_path, "w"), indent=1)
    json.dump(list(done.values()), open(out_path, "w"), indent=1)
    n_att = sum(1 for r in done.values() if (r.get("n_asserts") or 0) > 0)
    print(f"[assertions] {len(done)} | asserts inserted in {n_att} -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stage", required=True, choices=["conditions", "bodies", "assertions"])
    p.add_argument("--variant", default="refined", choices=list(VARIANTS),
                   help="condition-extraction prompt; see identify_contract_conditions.py")
    p.add_argument("--tag", default=DEFAULT_SETTING, help="setting name; names the output directory")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--api_key", default="")
    p.add_argument("--api_key_file", default="api_key.env")
    args = p.parse_args()
    tasks = load_contract_dataset("dataset/contracteval/test.json")
    from openai import OpenAI
    client = OpenAI(api_key=_load_key(args))
    {"conditions": stage_conditions, "bodies": stage_bodies,
     "assertions": stage_assertions}[args.stage](args, client, tasks)


if __name__ == "__main__":
    main()
