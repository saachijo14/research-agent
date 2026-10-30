from urllib.parse import urlparse

from agent.schemas import Claim, SubQuestion

# Tune these after you see real runs in the eval (Batch 5).
WEIGHTS = {"coverage": 0.30, "skeptic": 0.25, "tier_quality": 0.20, "diversity": 0.15, "volume": 0.10}
VERDICT_SCORE = {"survived": 1.0, "untested": 0.6, "weakened": 0.4, "contradicted": 0.0}
LOW_BELOW = 0.40
HIGH_FROM = 0.70


def is_primary(c: Claim) -> bool:
    return c.quote.startswith("GitHub API")


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def compute_confidence(claims: list[Claim], sub_questions: list[SubQuestion]) -> tuple[float, str, dict]:
    verified = [c for c in claims if c.verified]
    usable = [c for c in verified if c.verdict != "contradicted"]
    if not usable:
        return 0.0, "Low", {"usable_claims": 0}

    n_sub = max(len(sub_questions), 1)

    # 1) share of sub-questions with at least 2 usable claims
    coverage = sum(
        1 for sq in sub_questions if sum(1 for c in usable if c.sub_question_id == sq.id) >= 2
    ) / n_sub

    # 2) how well claims survived the Skeptic (primary API data counts as 1.0)
    skeptic = sum(1.0 if is_primary(c) else VERDICT_SCORE[c.verdict] for c in verified) / len(verified)

    # 3) share of usable claims from tier 1-2 sources
    tier_quality = sum(1 for c in usable if c.source_tier <= 2) / len(usable)

    # 4) number of independent domains (4+ = full marks)
    diversity = min(len({_domain(c.source_url) for c in usable}) / 4, 1.0)

    # 5) amount of evidence relative to the plan size
    volume = min(len(usable) / (3 * n_sub), 1.0)

    parts = {
        "coverage": coverage,
        "skeptic": skeptic,
        "tier_quality": tier_quality,
        "diversity": diversity,
        "volume": volume,
    }
    score = sum(WEIGHTS[k] * v for k, v in parts.items())

    web = [c for c in verified if not is_primary(c)]
    if web and sum(1 for c in web if c.verdict != "untested") / len(web) < 0.25:
        score = min(score, 0.69)   # too few web claims were actually challenged

    if len({c.source_url for c in usable}) < 2:
        score = min(score, 0.35)   # one source can never justify more than Low confidence

    # Hard caps: thin or lopsided evidence can never look confident.
    if len(usable) < 3:
        score = min(score, 0.35)
    if coverage < 0.5:
        score = min(score, 0.65)
    elif coverage < 0.8:
        score = min(score, 0.69)   # High requires nearly every sub-question to be covered

    score = round(score, 2)
    label = "Low" if score < LOW_BELOW else ("High" if score >= HIGH_FROM else "Medium")
    breakdown = {k: round(v, 2) for k, v in parts.items()}
    breakdown["usable_claims"] = len(usable)
    return score, label, breakdown