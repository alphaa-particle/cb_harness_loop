#!/bin/sh
set -e
cd /Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign
PY=.venv/bin/python
SCALE=data/external/scale_corpus/corpus.jsonl
E=scripts/evaluate_pipeline.py
$PY $E --split dev --runs fusion --search-methods fusion --label e2e_r5b_q_54 --set name_rules=2
$PY $E --questions usecases --split dev --runs fusion --search-methods fusion --label e2e_r5b_u_54 --set name_rules=2
$PY $E --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_r5b_q_scale --set name_rules=2
$PY $E --questions usecases --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_r5b_u_scale --set name_rules=2
echo R5B_DONE
