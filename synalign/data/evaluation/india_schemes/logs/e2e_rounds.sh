#!/bin/sh
# End-to-end answer check for each search round's cumulative state (dev only), after the baseline.
set -e
cd /Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign
while ! grep -q BASELINE_DONE data/evaluation/india_schemes/logs/e2e_r0.log; do sleep 10; done
PY=.venv/bin/python
SCALE=data/external/scale_corpus/corpus.jsonl
run() {  # label, settings...
  label=$1; shift
  $PY scripts/evaluate_pipeline.py --split dev --runs fusion --search-methods fusion --label e2e_${label}_q_54 --set "$@"
  $PY scripts/evaluate_pipeline.py --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_${label}_q_scale --set "$@"
  $PY scripts/evaluate_pipeline.py --questions usecases --split dev --runs fusion --search-methods fusion --label e2e_${label}_u_54 --set "$@"
  $PY scripts/evaluate_pipeline.py --questions usecases --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_${label}_u_scale --set "$@"
  echo "ROUND_DONE $label"
}
run r1 rules_count_once=true
run r2 rules_count_once=true named_schemes=true
run r3 rules_count_once=true named_schemes=true overview_last=true
echo ALL_ROUNDS_DONE
