"""Stage (ii) selection: example-signal argmax, then diff-logprob tie-break.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python code/utils/select_body.py <model> <hf_id> [setting]

Reads   bodies_greedy.json, bodies_sampled.json   the cap-5 candidate pool
        selected.json                             stage-1 result (select_by_examples.py)
Writes  selected.json                             updated in place
all under output/<setting>/<model>/.

Stage-1 pass counts are reused from `sel_scores`, so no candidate is executed twice. A task
reaches the tie-break only when two or more *distinct* candidates attain the top count. There,
each tied candidate is scored by the mean teacher-forced token log-probability over the lines
where the tied candidates differ (lines common to all of them carry no information), and the
highest score wins. Ties in the score keep the earlier candidate, so greedy is the fallback.
"""
import sys, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, "code")
from utils.paths import run_path, DEFAULT_SETTING


def norm_line(l):
    return "".join(l.split())


def main(S, hf, tag=DEFAULT_SETTING):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils.gen_common import build_chat_prompt
    from utils.instruction.baseline import _INSTRUCTION, _OUTPUT_INSTRUCTION
    from utils.dataset import load_contract_dataset

    greedy = {str(r["id"]): r for r in json.load(open(run_path(tag, S, "bodies_greedy")))}
    sampled = {str(r["id"]): r for r in json.load(open(run_path(tag, S, "bodies_sampled")))}
    view = json.load(open(run_path(tag, S, "view")))
    tasks = {str(t["id"]): t for t in load_contract_dataset("dataset/contracteval/test.json")}
    sel_path = run_path(tag, S, "selected")
    sel = json.load(open(sel_path))

    pool, ties = {}, {}
    for r in sel:
        tid = str(r["id"])
        cands = [(greedy.get(tid) or {}).get("code") or ""] + \
                list((sampled.get(tid) or {}).get("samples") or [])[:4]
        pool[tid] = cands
        scs = r.get("sel_scores")
        if not scs or len(scs) != len(cands) or tid not in view:
            continue
        mx = max(scs)
        T = [i for i, s in enumerate(scs) if s == mx]
        if len(T) > 1 and len(set(cands[i] for i in T)) > 1:
            ties[tid] = T
    print(f"[select-2] {S}: {len(ties)} tasks tie on the example count", flush=True)

    if ties:
        tok = AutoTokenizer.from_pretrained(hf, cache_dir=os.environ.get("HF_CACHE_DIR"),
                                            trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf, cache_dir=os.environ.get("HF_CACHE_DIR"),
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        system = f"{_INSTRUCTION['base']}\n\n{_OUTPUT_INSTRUCTION}"

        def token_lps(prompt, completion):
            """(log-prob, char offset) per completion token, by teacher forcing."""
            pi = tok(prompt, return_tensors="pt").input_ids
            enc = tok(completion, return_tensors="pt", add_special_tokens=False,
                      return_offsets_mapping=True)
            ids = torch.cat([pi, enc.input_ids], dim=1).to(model.device)
            with torch.no_grad():
                logits = model(ids).logits
            start = pi.shape[1] - 1
            lg = logits[0, start:-1].float()
            tgt = ids[0, start + 1:]
            sel_lp = (lg.gather(1, tgt.unsqueeze(1)).squeeze(1) - lg.logsumexp(dim=1)).cpu().tolist()
            return list(zip(sel_lp, [o[0] for o in enc.offset_mapping[0].tolist()]))

        picks, n_chg = {}, 0
        for tid, T in ties.items():
            cands = pool[tid]
            common = set.intersection(*[{norm_line(l) for l in cands[i].split("\n") if l.strip()}
                                        for i in T])
            t = tasks[tid]
            user = (f"Method Name: {t.get('entry_point') or tid}\n"
                    f"Problem Description:\n{view[tid]['masked'].strip()}")
            prompt = build_chat_prompt(tok, system, user, True)
            sc = {}
            for i in T:
                code = cands[i]
                lines = code.split("\n")
                line_of, pos = [], 0
                for ln, l in enumerate(lines):
                    line_of.append((pos, pos + len(l), ln))
                    pos += len(l) + 1
                diff_lines = {ln for ln, l in enumerate(lines)
                              if l.strip() and norm_line(l) not in common}
                if not diff_lines:                      # nothing distinctive -> cannot win
                    sc[i] = float("-inf"); continue
                dl = [v for v, cs in token_lps(prompt, code)
                      if any(a <= cs < b and ln in diff_lines for a, b, ln in line_of)]
                sc[i] = sum(dl) / len(dl) if dl else float("-inf")
            best = max(T, key=lambda i: (sc[i], -i))
            picks[tid] = best
            n_chg += best != min(T)
        print(f"[select-2] {S}: diff-logprob changed {n_chg} picks", flush=True)

        for r in sel:
            tid = str(r["id"])
            if tid in picks:
                r["code"] = pool[tid][picks[tid]]
                r["sel_pick"] = picks[tid]
                r["sel_stage"] = "diff-logprob"

    json.dump(sel, open(sel_path, "w"), indent=1)
    print(f"[select-2] {S}: wrote {len(sel)} -> {sel_path}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
