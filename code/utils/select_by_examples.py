"""Selection stage 1: rank the candidates by how many in-prompt example checks they pass.

Candidates are [greedy] + the temperature samples. Only the examples the model itself saw are
executed (artifacts/example_signals.json), so nothing held out enters the selection; a check the
reference solution fails is still used, because the model read the same example. Ties keep the
earlier candidate, which makes greedy the fallback when there is no signal at all.

    python code/utils/select_by_examples.py <greedy.json> <sampled.json> <selected.json>

The output keeps the input schema and adds the sel_* diagnostic fields.
"""
import json, sys, os, io, contextlib, multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
def _load(name):
    """Read an artifacts/ file on first use, with a pointer to the script that rebuilds it."""
    import json as _json, os as _os
    p = _os.path.join("artifacts", name)
    if not _os.path.exists(p):
        raise FileNotFoundError(f"{p} is missing — run scripts/build_artifacts.sh to rebuild it")
    return _json.load(open(p, encoding="utf-8"))


_CHECKS = _PROSE = None


def signals(tid):
    """Example checks for a task: the parsed ones, else the prose fallback."""
    global _CHECKS, _PROSE
    if _CHECKS is None:
        _CHECKS = _load("example_signals.json")
        _PROSE = _load("prose_examples.json")     # tasks with no parsable check
    return [c["stmt"] for c in _CHECKS.get(tid, {}).get("checks", [])] or _PROSE.get(tid) or []


def run_checks(code, stmts, timeout=8.0):
    """Load `code` and return one pass/fail per check; a failure or timeout is all False."""
    def worker(q):
        env = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, env)
                flags = []
                for s in stmts:
                    try:
                        exec(s, env); flags.append(True)
                    except Exception:
                        flags.append(False)
            q.put(flags)
        except Exception:
            q.put([False] * len(stmts))

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return [False] * len(stmts)
    try:
        return q.get_nowait()
    except Exception:
        return [False] * len(stmts)


def main(greedy_path, samples_path, out_path):
    greedy = {str(r["id"]): r for r in json.load(open(greedy_path))}
    sampled = {str(r["id"]): r for r in json.load(open(samples_path))}
    out, stats = [], {"no_signal": 0, "kept_greedy": 0, "switched": 0}
    for tid, g in greedy.items():
        checks = signals(tid)
        s = sampled.get(tid) or {}
        cands = [g.get("code") or ""] + list(s.get("samples") or [])
        rec = dict(g)
        if not checks or len(cands) == 1:
            stats["no_signal"] += 1
            rec.update(sel_signal=0, sel_pick=0)
        else:
            scores = [sum(run_checks(c, checks)) for c in cands]
            best = max(range(len(cands)), key=lambda i: (scores[i], -i))
            rec.update(code=cands[best], sel_signal=len(checks),
                       sel_pick=best, sel_scores=scores)
            stats["kept_greedy" if best == 0 else "switched"] += 1
        out.append(rec)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[select-1] {len(out)} tasks | no signal, greedy kept {stats['no_signal']} | "
          f"greedy won {stats['kept_greedy']} | replaced by a sample {stats['switched']} "
          f"-> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
