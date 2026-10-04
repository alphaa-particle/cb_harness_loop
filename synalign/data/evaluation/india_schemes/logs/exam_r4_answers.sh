#!/bin/sh
# Round 4 alone (name_rules=1) on the exam sets it was not yet run on, after the main exam finishes.
set -e
cd /Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign
while ! grep -q EXAM_DONE data/evaluation/india_schemes/logs/final_exam_2.log; do sleep 15; done
PY=.venv/bin/python
SCALE=data/external/scale_corpus/corpus.jsonl
E=scripts/evaluate_pipeline.py
$PY $E --split test --runs fusion --search-methods fusion --corpus $SCALE --label exam_r4_after_questions_scale --set name_rules=1
$PY $E --questions usecases --split test --runs fusion --search-methods fusion --label exam_r4_after_usecases_54 --set name_rules=1
$PY $E --questions usecases --split test --runs fusion --search-methods fusion --corpus $SCALE --label exam_r4_after_usecases_scale --set name_rules=1
echo R4_EXAM_DONE
