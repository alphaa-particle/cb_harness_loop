# Design and diagnosis records for the search cycle

Saved so every number in [SYNALIGN_RETRIEVAL_ROUNDS.md](../../../../docs/results/SYNALIGN_RETRIEVAL_ROUNDS.md)
that is not in `search_rounds/` or an `e2e_*` folder has a source. All measured on practice (dev)
questions only.

| File | What it is |
|---|---|
| `diagnosis_r0_dev_54.jsonl`, `diagnosis_r0_dev_scale.jsonl` | every dev question before the cycle: the prompt it got, the top 20 ranked sections with meaning scores, and why it missed if it did. The `cause` field: `wrong_scheme` = the asked scheme is not among the schemes in the prompt; `crowded_out` = it is, but not the section holding the answer; `not_found` = nothing retrieved |
| `design1_proposals_and_review.json` | the four designers' proposals (every variant they tried, with numbers) and the reviewer's verdicts |
| `design1_fusion_*.log` | the meaning-over-words designer's runs |
| `design1_scheme_level_results_*.json` | the scheme-first designer's runs (about 35 variants) |
| `design1_representation_*.json` | the representation designer's runs (including the romaniser that was set aside) |
| `design1_abstention_*.json` | the "not found" designer's 22 rules |
| `design1_review_*.json` | the reviewer's confirmations, combinations and forecast |
| `prototype_evidence_assembly.json` | the first prototype of the three evidence mechanisms (rounds 1-3), all combinations |
| `prototype_evidence_assembly_guards.json` | the same with the two name guards (unique names, top scheme kept) |
