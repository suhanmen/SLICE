#!/bin/bash
# =============================================================================
# Rebuild artifacts/ from the dataset. No model, no GPU, no network — every step is
# deterministic, so the result is the same on any machine.
#
#   cd copy_code && sh scripts/build_artifacts.sh
#
# example_signals/prose_examples come from the task descriptions; the other three are the
# reference-contract artifacts the comparison arms need. prose_examples consumes
# example_signals, and condition_specs consumes condition_nodes.
#   condition_nodes.json      reference contract text -> individual conditions (comparison arms)
#   functional_view_base.json specification graph on the reference contract text (comparison arm)
#   example_signals.json      executable examples, read out of the task descriptions
#   prose_examples.json       examples written in prose, for the tasks with no parsed check
#   condition_specs.json      conditions -> structured specs (template comparison arm)
#
# condition_specs.json is built only when it is absent. If a copy is already there it is the
# frozen file the reported runs used, and rebuilding would replace it with a superset (the
# condition set grew after it was frozen), so --with-specs writes to condition_specs_new.json
# instead and leaves the frozen file alone.
# =============================================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=code
PY=${PYTHON:-python}
mkdir -p artifacts

echo "--- 1/4+ condition_nodes.json"
$PY code/utils/condition_decompose.py > /dev/null

echo "--- 2/4+ functional_view_base.json  (a minute or so)"
$PY code/utils/spec_graph.py > /dev/null

echo "--- 3/4+ example_signals.json  (executes the reference solution; several minutes)"
$PY code/utils/example_signals.py | tail -3

echo "--- 4/4+ prose_examples.json"
$PY code/utils/prose_examples.py | grep recovered

if [ ! -f artifacts/condition_specs.json ]; then
  echo "--- 5/5 condition_specs.json  (comparison arms)"
  $PY code/utils/parse_conditions.py artifacts/condition_specs.json | grep -E "parsed|unparsed"
elif [ "$1" = "--with-specs" ]; then
  echo "--- extra: condition_specs_new.json (the frozen condition_specs.json is left alone)"
  $PY code/utils/parse_conditions.py artifacts/condition_specs_new.json | grep -E "parsed|unparsed"
fi

rm -f artifacts/spec_graphs.json artifacts/graph_mask_report.md artifacts/split_report.md
echo "=== done"
ls -la artifacts/
