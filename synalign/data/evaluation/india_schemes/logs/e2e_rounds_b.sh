#!/bin/sh
# End-to-end answer check for rounds 2b and 3b (round 1 was undone at its answer check).
set -e
cd /Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign
PY=.venv/bin/python
SCALE=data/external/scale_corpus/corpus.jsonl
run() {
  label=$1; shift
  $PY scripts/evaluate_pipeline.py --split dev --runs fusion --search-methods fusion --label e2e_${label}_q_54 --set "$@"
  $PY scripts/evaluate_pipeline.py --questions usecases --split dev --runs fusion --search-methods fusion --label e2e_${label}_u_54 --set "$@"
  $PY scripts/evaluate_pipeline.py --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_${label}_q_scale --set "$@"
  $PY scripts/evaluate_pipeline.py --questions usecases --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_${label}_u_scale --set "$@"
  echo "ROUND_DONE $label"
}
run r2b named_schemes=true
run r3b named_schemes=true overview_last=true
echo ALL_ROUNDS_DONE
