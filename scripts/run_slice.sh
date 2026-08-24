#!/bin/bash
# =============================================================================
# SLICE — full chain for one local HF model, stages (i) -> (ii) -> (iii) -> scoring.
#
#   cd copy_code && sh scripts/run_slice.sh <model_short> <hf_id> [GPU] [setting]
#   e.g.  sh scripts/run_slice.sh Qwen3.5-9B Qwen/Qwen3.5-9B 0
#
# <model_short> must be the last path segment of <hf_id> — the stages name their files after it.
#
# Every generated file goes to output/<setting>/<model_short>/, so running a second setting on the
# same model never overwrites the first. <setting> defaults to SLICE (the method).
#
# Final code    : output/<setting>/<model_short>/final.json
# Per-task score: evaluation/contracteval/<model_short>/<setting>__breakdown.json
# =============================================================================
set -e
S=${1:?model_short (e.g. Qwen3.5-9B)}
M=${2:?hf id (e.g. Qwen/Qwen3.5-9B)}
GPU=${3:-0}
TAG=${4:-SLICE}

# ---- parameters: changing these changes the method -------------------------
CAP=5                   # body candidates = 1 greedy + 4 at T=0.7; efficiency frontier of a 9->7->5 search
TEMP=0.7                # sampling temperature for the diverse candidates
MAX_NEW=2048            # generation budget, shared by every stage
NO_ASSUME=True          # "may assume" clauses state an assumption, so they are not enforced
# The hybrid anchor thresholds (TAU 0.75 / FLOOR 0.45) live in code/utils/functional_view.py.
# A reasoning model that cannot turn thinking off spends the budget on thinking tokens:
#   THINK_PREFILL=1 sh scripts/run_slice.sh <model_short> <hf_id>
# Model cache location (optional): export HF_CACHE_DIR=/path/to/hf/hub
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
export PYTHONPATH=code
PY=${PYTHON:-python}
D="output/$TAG/$S"
G="CUDA_VISIBLE_DEVICES=$GPU"

echo "=== [$S] SLICE setting=$TAG (GPU $GPU, cap$CAP, T$TEMP) -> $D ==="

echo "--- (i) contract conditions (LLM, greedy)"
eval $G $PY code/identify_contract_conditions.py --model_name "$M" --variant refined --tag "$TAG"

echo "--- (i) specification graph -> functional view (embedding forward, no generation)"
eval $G $PY code/utils/functional_view.py "$S" "$TAG"

echo "--- (i) example signals (deterministic, model-independent; skipped if already present)"
# The only two artifacts the method needs; both are rebuilt here if artifacts/ was cleared.
[ -f artifacts/example_signals.json ] || $PY code/utils/example_signals.py
[ -f artifacts/prose_examples.json ] || $PY code/utils/prose_examples.py

echo "--- (ii) $CAP body candidates (1 greedy + $((CAP-1)) at T$TEMP)"
eval $G $PY code/generate_bodies.py --model_name "$M" --tag "$TAG" --limit 340 \
  --max_new_tokens $MAX_NEW --out_tag greedy --resume True
eval $G $PY code/generate_bodies.py --model_name "$M" --tag "$TAG" --limit 340 \
  --max_new_tokens $MAX_NEW --out_tag sampled \
  --n_samples $((CAP-1)) --temperature $TEMP --resume True

echo "--- (ii) selection: example-count argmax, then diff-logprob on ties"
$PY code/utils/select_by_examples.py "$D/bodies_greedy.json" "$D/bodies_sampled.json" \
  "$D/selected.json"
eval $G $PY code/utils/select_body.py "$S" "$M" "$TAG"

echo "--- (iii) contract assertions (one per condition)"
eval $G $PY code/generate_assertions.py --model_name "$M" --tag "$TAG" --resume True \
  --no_assume $NO_ASSUME --max_new_tokens $MAX_NEW

echo "--- (iii) AST post-processing (argument renaming -> ordering -> dedup -> insertion)"
$PY code/utils/rename_and_insert.py "$D/assertions_raw.json" "$D/selected.json" "$D/inserted.json"

echo "--- (iii) screening (drop only assertions that reject a known-valid example)"
$PY code/utils/screen_assertions.py "$D/inserted.json" "$D/final.json"

echo "--- scoring (lenient and strict in one pass, per-task store)"
$PY code/utils/results_store.py score "$TAG" --models "$S"

echo "=== [$S/$TAG] done -> $D/final.json ==="
