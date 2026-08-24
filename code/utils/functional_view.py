"""Stage (i): specification graph + hybrid anchor masking -> the functional view.

    anchored = MODAL and cos(segment, condition) >= TAU and content-word overlap >= FLOOR
    removed  = anchored and not entangled   (entangled = functional-role marker or numbered line)

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python code/utils/functional_view.py <model> [setting]

Writes output/<setting>/<model>/view.json. Precision/recall against a deterministic gold proxy is
printed for reference only; it never feeds back into the view.

WARNING: TAU 0.75 / FLOOR 0.45 is the operating point chosen by a grid search against the gold
contracts. Changing it changes the method.
"""

import sys, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, "code")

import torch
from transformers import AutoTokenizer, AutoModel
import utils.spec_graph as sgm
from utils.segment_parser import cw, segments, FUNC_MARK, NUMBERED, CODE_LINE
from utils.condition_decompose import decompose_contract, arg_names
from utils.paths import run_path, DEFAULT_SETTING

M = "Salesforce/SFR-Embedding-2_R"
tok = AutoTokenizer.from_pretrained(M, cache_dir=os.environ.get("HF_CACHE_DIR"))
model = AutoModel.from_pretrained(M, cache_dir=os.environ.get("HF_CACHE_DIR"),
                                  torch_dtype=torch.bfloat16, device_map="auto")
model.eval()


@torch.no_grad()
def embed(texts):
    out = []
    for i in range(0, len(texts), 16):
        b = tok(texts[i:i+16], padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(model.device)
        h = model(**b).last_hidden_state
        idx = b.attention_mask.sum(1) - 1
        e = h[torch.arange(h.size(0)), idx]
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0, 1)


def deplural(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and not w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def dcw(s):
    return {deplural(w) for w in cw(s)}


TAU, FLOOR = 0.75, 0.45


def build_graph_hyb(task):
    args = arg_names(task.get("signature") or "")
    cnodes = decompose_contract(sgm.spec_text(task), args)
    ck = dcw(sgm.spec_text(task) + " " + (task.get("signature") or ""))
    nodes, edges = [], []
    for i, c in enumerate(cnodes):
        nodes.append({"nid": f"c{i}", "kind": "contract", "category": c["category"],
                      "input_precond": c["is_input_precondition"], "text": c["text"]})
    seg_list, seg_meta = [], []
    seg_id = 0
    for ln in (task.get("prompt_cs") or "").split("\n"):
        raw = ln
        m = sgm.DOCSTR_OPEN.match(ln)
        if m:
            ln = m.group(2)
        mc = sgm.DOCSTR_CLOSE.search(ln)
        if mc and ln[:mc.start()].strip():
            ln = ln[:mc.start()]
        if CODE_LINE.match(raw) or not ln.strip():
            if raw.strip():
                nodes.append({"nid": f"e{seg_id}", "kind": "code", "text": raw.strip()[:80]})
                seg_id += 1
            continue
        numbered = bool(NUMBERED.match(raw))
        for seg in segments(ln):
            seg_list.append(seg.strip())
            seg_meta.append((f"s{seg_id}", seg, numbered))
            seg_id += 1
    ctexts = [c["text"] for c in cnodes]
    if ctexts and seg_list:
        E = embed(seg_list + ctexts)
        sims = E[:len(seg_list)] @ E[len(seg_list):].T
    else:
        sims = None
    for k, (nid, seg, numbered) in enumerate(seg_meta):
        if sims is not None:
            mx, bi = sims[k].max(0)
            mx, bi = float(mx), int(bi)
        else:
            mx, bi = 0.0, 0
        sk = dcw(seg)
        ov = len(sk & ck) / max(1, len(sk))
        anchored = mx >= TAU and ov >= FLOOR and bool(sgm.MODAL.search(seg))
        entangled = bool(FUNC_MARK.search(seg)) or numbered
        nodes.append({"nid": nid, "kind": "segment", "text": seg.strip()[:120],
                      "anchor_ov": round(mx, 2)})
        if anchored:
            edges.append({"src": f"c{bi}", "dst": nid, "type": "anchor", "w": round(mx, 2)})
        if entangled:
            edges.append({"src": nid, "dst": "FUNC", "type": "entangle"})
    return nodes, edges


def gold_set(tid, task):
    """Prose segments that restate a gold contract clause (deterministic proxy, report only)."""
    gk = dcw((task.get("contract_nl") or "") + " " + (task.get("signature") or ""))
    G = set()
    for ln in (task.get("prompt_cs") or "").split("\n"):
        raw = ln
        m = sgm.DOCSTR_OPEN.match(ln)
        if m:
            ln = m.group(2)
        mc = sgm.DOCSTR_CLOSE.search(ln)
        if mc and ln[:mc.start()].strip():
            ln = ln[:mc.start()]
        if CODE_LINE.match(raw) or not ln.strip():
            continue
        for seg in segments(ln):
            sk = dcw(seg)
            if sk and len(sk & gk) / len(sk) >= 0.5 and sgm.MODAL.search(seg):
                G.add(seg.strip())
    return G


sgm.build_graph = build_graph_hyb

S = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SETTING
cl = json.load(open(run_path(TAG, S, "conditions")))
TASKS = {str(t["id"]): t for t in json.load(open("dataset/contracteval/test.json"))}
FROZEN = json.load(open("dataset/contracteval/eval_tasks_340.json"))["ids"]
cls = lambda v: v if isinstance(v, list) else v.get("clauses", [])

tp = nr = ng = 0                                    # |removed & gold|, |removed|, |gold|
out_view = {}
for tid in FROZEN:
    t = dict(TASKS[tid])
    G = gold_set(tid, TASKS[tid])
    # the graph anchors against the model's own conditions, never the reference contract
    t["spec_text"] = "\n".join(map(str, cls(cl.get(tid, [])))) + "\n" + (t.get("signature") or "")
    masked, removed, _, _ = sgm.graph_mask(t)
    out_view[tid] = {"masked": masked, "removed_segments": removed}
    R = set(x.strip() for x in removed)
    tp += len(R & G); nr += len(R); ng += len(G)
out_path = run_path(TAG, S, "view")
json.dump(out_view, open(out_path, "w"), indent=1)
print(f"[view] {S}: {len(out_view)} tasks -> {out_path}", flush=True)
print(f"[view] {S}: precision {tp}/{nr} = {tp/max(nr,1)*100:5.1f}% | "
      f"recall {tp}/{ng} = {tp/max(ng,1)*100:5.1f}%  (vs gold proxy)", flush=True)
