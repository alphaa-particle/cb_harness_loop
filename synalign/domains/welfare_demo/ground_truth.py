"""Ground-truth builder for the welfare demo domain.

Contract with the engine (same for every domain):

    build_ground_truth(profile: dict, visible: dict) -> dict

`profile`  - the full synthetic user (some values may be None = user doesn't know).
`visible`  - only the fields the question actually revealed.

The returned dict must contain these standard keys:
    likely_eligible              list[str]   entity IDs a good answer presents as likely
    unknown_due_to_missing_info  list[str]   entity IDs that cannot be confirmed yet
    ineligible                   list[str]   entity IDs a good answer rules out
    must_ask_about               list[str]   FIELD names the assistant should ask about
    must_not_claim               list[str]   forbidden-claim IDs (from aliases.yaml)
    ineligibility_reasons        dict        entity ID -> why it is ruled out (optional; used in training targets)
    gold_chunk_ids               list[str]   document sections a correct answer needs
    ideal_behavior               str         one-sentence behavioral spec
"""


def _pmsym(visible: dict) -> tuple[str, list[str], list[str]]:
    """Return ('yes'|'no'|'unknown', missing_fields, reasons) for PMSYM, using ONLY visible info."""
    needed = ["age", "income", "worker_type", "epfo"]
    missing = [f for f in needed if visible.get(f) is None]

    # If any visible field already disqualifies, the verdict is 'no'
    # even when other fields are missing.
    reasons = []
    if visible.get("age") is not None and not (18 <= visible["age"] <= 40):
        reasons.append("age is outside the 18–40 range")
    if visible.get("income") is not None and visible["income"] > 15000:
        reasons.append("monthly income is above the ₹15,000 limit")
    if visible.get("worker_type") is not None and visible["worker_type"] != "unorganised_worker":
        reasons.append("PMSYM is for unorganised workers")
    if visible.get("epfo") is True:
        reasons.append("EPFO-covered applicants are not eligible")
    if reasons:
        return "no", [], reasons

    if missing:
        return "unknown", missing, []
    return "yes", [], []


def _eshram(visible: dict) -> tuple[str, list[str], list[str]]:
    needed = ["age", "worker_type"]
    missing = [f for f in needed if visible.get(f) is None]

    reasons = []
    if visible.get("age") is not None and not (16 <= visible["age"] <= 59):
        reasons.append("age is outside the 16–59 range")
    if visible.get("worker_type") is not None and visible["worker_type"] not in (
        "unorganised_worker", "self_employed"
    ):
        reasons.append("eShram is for unorganised or self-employed workers")
    if reasons:
        return "no", [], reasons

    if missing:
        return "unknown", missing, []
    return "yes", [], []


def build_ground_truth(profile: dict, visible: dict) -> dict:
    pmsym_verdict, pmsym_missing, pmsym_reasons = _pmsym(visible)
    eshram_verdict, eshram_missing, eshram_reasons = _eshram(visible)

    verdicts = {"scheme_pmsym": pmsym_verdict, "scheme_eshram": eshram_verdict}
    missing_by_scheme = {"scheme_pmsym": pmsym_missing, "scheme_eshram": eshram_missing}

    likely = [s for s, v in verdicts.items() if v == "yes"]
    unknown = [s for s, v in verdicts.items() if v == "unknown"]
    ineligible = [s for s, v in verdicts.items() if v == "no"]

    # The assistant should ask about every field that blocks a verdict.
    must_ask_about = sorted({f for s in unknown for f in missing_by_scheme[s]})

    must_not_claim = ["guaranteed_approval", "automatic_cash_benefit"]
    reasons = {"scheme_pmsym": "; ".join(pmsym_reasons), "scheme_eshram": "; ".join(eshram_reasons)}

    # In this small demo both schemes are always relevant context.
    gold_chunk_ids = ["scheme_pmsym", "scheme_eshram"]

    ideal_behavior = (
        f"Present {likely or 'no scheme'} as likely eligible, "
        f"present {unknown or 'none'} as needing confirmation, "
        f"ask about {must_ask_about or 'nothing'}, "
        "never guarantee approval, and ground every claim in the documents."
    )

    return {
        "likely_eligible": likely,
        "unknown_due_to_missing_info": unknown,
        "ineligible": ineligible,
        "must_ask_about": must_ask_about,
        "must_not_claim": must_not_claim,
        "ineligibility_reasons": {s: reasons[s] for s in ineligible},
        "gold_chunk_ids": gold_chunk_ids,
        "ideal_behavior": ideal_behavior,
    }
