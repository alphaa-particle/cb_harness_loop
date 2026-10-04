#!/bin/sh
# The exam, run once after the last round, on the kept settings (eval_config.yaml) against the
# settings before this cycle. Never used to choose or undo anything.
set -e
cd /Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign
PY=.venv/bin/python
SCALE=data/external/scale_corpus/corpus.jsonl
OFF="add_named_schemes=false"
E=scripts/evaluate_pipeline.py
# search alone, before and after, both question sets, both sizes
for Q in questions usecases; do
  $PY $E --questions $Q --split test --search-only --search-methods fusion --label exam_search_after_${Q}_54
  $PY $E --questions $Q --split test --search-only --search-methods fusion --label exam_search_before_${Q}_54 --set $OFF
  $PY $E --questions $Q --split test --search-only --search-methods fusion --corpus $SCALE --label exam_search_after_${Q}_scale
  $PY $E --questions $Q --split test --search-only --search-methods fusion --corpus $SCALE --label exam_search_before_${Q}_scale --set $OFF
done
echo SEARCH_DONE
# answers: original exam questions after (before = test_final / test_scale)
$PY $E --split test --runs fusion --search-methods fusion --label exam_after_questions_54
$PY $E --split test --runs fusion --search-methods fusion --corpus $SCALE --label exam_after_questions_scale
# answers: use-case exam questions, before and after; reference conditions at 54
$PY $E --questions usecases --split test --runs closed_book rules_only fusion oracle --search-methods fusion --label exam_after_usecases_54
$PY $E --questions usecases --split test --runs fusion --search-methods fusion --label exam_before_usecases_54 --set $OFF
$PY $E --questions usecases --split test --runs fusion --search-methods fusion --corpus $SCALE --label exam_after_usecases_scale
$PY $E --questions usecases --split test --runs fusion --search-methods fusion --corpus $SCALE --label exam_before_usecases_scale --set $OFF
echo EXAM_DONE
