#!/bin/sh
set -e
cd /Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign
PY=.venv/bin/python
SCALE=data/external/scale_corpus/corpus.jsonl
E=scripts/evaluate_pipeline.py
$PY $E --split dev --runs fusion --search-methods fusion --label e2e_r4_q_54 --set add_named_schemes=true
$PY $E --questions usecases --split dev --runs fusion --search-methods fusion --label e2e_r4_u_54 --set add_named_schemes=true
$PY $E --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_r4_q_scale --set add_named_schemes=true
$PY $E --questions usecases --split dev --runs fusion --search-methods fusion --corpus $SCALE --label e2e_r4_u_scale --set add_named_schemes=true
echo R4_DONE
