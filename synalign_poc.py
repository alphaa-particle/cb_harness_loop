"""
SYNALIGN — proof-of-concept audit harness
=======================================
Linguistic dependencies REMOVED. Goal reframed: improve a chatbot's general
performance (robustness + behavioral quality), not cross-language equity.

What changed vs the language version
------------------------------------
  - No languages, no translate-pivot, no multilingual judge, no Indic anchors.
  - The counterfactual axis is now INPUT ROBUSTNESS: hold the persona +
    eligibility truth fixed, and vary how the QUESTION is posed
    (clean / vague / missing-info / typo-heavy / misleading). Eligibility is
    invariant to phrasing, so any score drop is a pure system-behavior failure.
  - Plus an optional eligibility-IRRELEVANT attribute perturbation (a fairness
    invariance check) that also needs no language.
  - Because every remaining gap is BEHAVIORAL, it is alignment-fixable. We
    simulate an SFT+KTO-improved assistant and re-audit to show the gap closes
    (the language gap could not be closed this way).

Ground truth is still the deterministic rule engine, so correctness is
judge-free. Stats still carry bootstrap CIs + Holm correction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

RNG = np.random.default_rng(42)

# --------------------------------------------------------------------------- #
# 1. Population = scenario fixtures (demographic realism matters far less now;
#    coverage of the TASK/EDGE space is what matters for performance).
# --------------------------------------------------------------------------- #
def make_population(n=4000):
    age = RNG.integers(18, 65, n)
    income = np.clip(RNG.normal(13000, 4500, n), 3000, 60000).round(-2)
    worker_type = RNG.choice(["unorganised_worker", "salaried", "self_employed"],
                             n, p=[0.6, 0.15, 0.25])
    epfo = (worker_type == "salaried") & (RNG.random(n) < 0.8)
    literacy = RNG.choice(["low", "medium", "high"], n, p=[0.35, 0.4, 0.25])
    return pd.DataFrame(dict(age=age, income=income, worker_type=worker_type,
                             epfo=epfo, literacy=literacy))

# --------------------------------------------------------------------------- #
# 2. Deterministic rule engine = objective, judge-free ground truth.
# --------------------------------------------------------------------------- #
@dataclass
class Scheme:
    rule_id: str; age_min: int; age_max: int; income_max: int
    worker_types: tuple; excludes_epfo: bool

PMSYM = Scheme("scheme_pmsym", 18, 40, 15000, ("unorganised_worker",), True)
ESHRAM = Scheme("scheme_eshram", 16, 59, 999999, ("unorganised_worker", "self_employed"), False)
SCHEMES = [PMSYM, ESHRAM]

def eligible(row, s):
    return (s.age_min <= row.age <= s.age_max and row.income <= s.income_max
            and row.worker_type in s.worker_types
            and not (s.excludes_epfo and row.epfo))

def ground_truth(row):
    return {s.rule_id for s in SCHEMES if eligible(row, s)}

# --------------------------------------------------------------------------- #
# 3. Robustness axis (replaces language). Eligibility truth is INVARIANT to
#    these phrasing conditions, so any performance drop is a behavioral failure.
# --------------------------------------------------------------------------- #
CONDITIONS = ["clean", "vague", "missing_info", "typo_heavy", "misleading"]
# how hard each condition is for a baseline chatbot (1.0 = no degradation)
COND_DIFFICULTY = {"clean": 1.0, "vague": 0.78, "missing_info": 0.62,
                   "typo_heavy": 0.7, "misleading": 0.5}

# --------------------------------------------------------------------------- #
# 4. Assistants. Baseline degrades under harder phrasings (and is the failure
#    we want to fix). The "aligned" assistant has been SFT+KTO-trained to ask
#    follow-ups and resist misleading framing -> recovers most of the gap.
#    Swap either for a real RAG+rule pipeline; the harness is unchanged.
# --------------------------------------------------------------------------- #
class Assistant:
    def __init__(self, robustness=0.0, halluc_rate=0.06):
        # robustness in [0,1]: 0 = baseline, ~1 = fully hardened behavior
        self.robustness = robustness
        self.halluc_rate = halluc_rate
    def answer(self, row, condition):
        truth = ground_truth(row)
        d = COND_DIFFICULTY[condition]
        # alignment lifts the effective difficulty floor toward 'clean'
        d_eff = d + (1 - d) * self.robustness
        said = {g for g in truth if RNG.random() < d_eff}
        # misleading framing induces hallucination; alignment suppresses it
        h = self.halluc_rate * (2.0 - d) * (1 - 0.8 * self.robustness)
        if RNG.random() < h:
            said.add("scheme_pmsym" if "scheme_pmsym" not in truth else "scheme_eshram")
        return said

# --------------------------------------------------------------------------- #
# 5. Scoring vs objective truth.
# --------------------------------------------------------------------------- #
def score(said, truth):
    if truth:
        recall = len(said & truth) / len(truth)
    else:
        recall = 1.0 if not (said - truth) else 0.0
    hallucinated = 1.0 if (said - truth) else 0.0
    return recall, hallucinated

def run_audit(pop, assistant):
    rows = []
    for _, p in pop.iterrows():
        truth = ground_truth(p)
        for cond in CONDITIONS:                      # counterfactual: same persona, vary phrasing
            said = assistant.answer(p, cond)
            rec, hal = score(said, truth)
            rows.append(dict(condition=cond, literacy=p.literacy,
                             recall=rec, hallucination=hal))
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# 6. Gap stats: bootstrap 95% CI + Holm-corrected p-values.
# --------------------------------------------------------------------------- #
def bootstrap_gap(ref, grp, B=4000):
    ref, grp = np.asarray(ref), np.asarray(grp)
    obs = ref.mean() - grp.mean()
    boot = np.array([RNG.choice(ref, len(ref), True).mean()
                     - RNG.choice(grp, len(grp), True).mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    _, p = stats.ttest_ind(ref, grp, equal_var=False)
    return obs, lo, hi, p

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); prev = 0.0
    for rank, idx in enumerate(order):
        prev = max(prev, min((m - rank) * pvals[idx], 1.0)); adj[idx] = prev
    return adj

def gap_report(audit, metric="recall", ref="clean"):
    refv = audit.loc[audit.condition == ref, metric].values
    recs = []
    for c in [c for c in CONDITIONS if c != ref]:
        g = audit.loc[audit.condition == c, metric].values
        obs, lo, hi, p = bootstrap_gap(refv, g)
        recs.append(dict(condition=c, n=len(g), ref_mean=round(refv.mean(), 3),
                         grp_mean=round(g.mean(), 3), gap=round(obs, 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3), p_raw=p))
    df = pd.DataFrame(recs)
    df["p_holm"] = holm(df["p_raw"].values)
    df["significant"] = (df["p_holm"] < 0.05) & (df["ci_lo"] > 0)
    df["p_raw"] = df["p_raw"].round(4); df["p_holm"] = df["p_holm"].round(4)
    return df.sort_values("gap", ascending=False)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pop = make_population(4000)
    base = Assistant(robustness=0.0)
    aligned = Assistant(robustness=0.75)             # after SFT + KTO on failures

    a_base = run_audit(pop, base)
    a_algn = run_audit(pop, aligned)

    print("=== Eligibility-IRRELEVANT attribute check (literacy) — should be ~0 ===")
    lit = a_base.groupby("literacy").recall.mean()
    print(lit.round(3).to_string())
    print(f"max literacy spread = {lit.max()-lit.min():.3f} (flag if material)\n")

    print("=== BASELINE: recall gap vs 'clean' phrasing (CI + Holm) ===")
    print(gap_report(a_base, "recall").to_string(index=False))
    print(f"\nbaseline overall recall = {a_base.recall.mean():.3f} | "
          f"worst-condition recall = {a_base.groupby('condition').recall.mean().min():.3f} | "
          f"halluc = {a_base.hallucination.mean():.3f}")

    print("\n=== AFTER SFT+KTO: same audit, frozen fixtures ===")
    print(gap_report(a_algn, "recall").to_string(index=False))
    print(f"\naligned overall recall  = {a_algn.recall.mean():.3f} | "
          f"worst-condition recall = {a_algn.groupby('condition').recall.mean().min():.3f} | "
          f"halluc = {a_algn.hallucination.mean():.3f}")

    print("\n=== Before/after (the deliverable claim) ===")
    b = a_base.groupby("condition").recall.mean()
    al = a_algn.groupby("condition").recall.mean()
    comp = pd.DataFrame({"baseline": b.round(3), "aligned": al.round(3),
                         "change": (al - b).round(3)}).sort_values("change", ascending=False)
    print(comp.to_string())
    print(f"\nworst-condition recall: {b.min():.3f} -> {al.min():.3f} "
          f"(+{al.min()-b.min():.3f}); hallucination "
          f"{a_base.hallucination.mean():.3f} -> {a_algn.hallucination.mean():.3f}")
