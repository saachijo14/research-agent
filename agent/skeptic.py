import time
from datetime import date
import re

from agent.extractor import extract_claims
from agent.llm import llm_json
from agent.schemas import Claim, CounterQuery, SkepticJudgement
from agent.tools.registry import call_tool
from agent.verifier import sanitize_text, select_urls, verify_claims

MAX_CLAIMS_TO_CHALLENGE = 5   # cost control: challenge only the top claims
MAX_TOOL_CALLS = 12            # the Skeptic's own budget, separate from the executor's
MAX_SECONDS = 150

ABSOLUTE = re.compile(
    r"\b(only|lowest|highest|fastest|slowest|best|worst|largest|cheapest|always|never|requires?|must|all three)\b",
    re.IGNORECASE,
)

QUERY_SYSTEM = f"""You are the Skeptic in a research pipeline. Today's date is {date.today().isoformat()}.
Given a claim, write ONE web search query (max 12 words) that would surface evidence AGAINST it:
limitations, problems, deprecation, newer information, or a competing view.
Name the specific entity from the claim. Do not restate the claim as a positive statement."""

JUDGE_SYSTEM = """You are the Skeptic in a research pipeline. You get a CLAIM and COUNTER-EVIDENCE:
short verified quotes found by searching for evidence against the claim.
Decide:
- "contradicted": the counter-evidence states that the specific fact in the claim is FALSE for the same
  entity (for example a different license, a shutdown, or a different number). Merely mentioning a commercial
  edition, a limitation, or a competing view is NOT enough.
- "weakened": it adds an important caveat, limitation, newer information, or a conflicting but not decisive view.
- "survived": it is unrelated, vague, or supports the claim.
Be conservative: unrelated or vague counter-evidence means "survived". Use ONLY the evidence given,
no outside knowledge. note: at most 2 sentences saying what you found."""


def _is_primary(c: Claim) -> bool:
    return c.quote.startswith("GitHub API")  # API data is not opinion; nothing to challenge


def pick_targets(claims: list[Claim], n: int) -> list[Claim]:
    """Round-robin across sub-questions so one topic doesn't use all the Skeptic's budget."""
    by_sub: dict[str, list[Claim]] = {}
    for c in claims:
        if c.verified and not _is_primary(c):
            by_sub.setdefault(c.sub_question_id, []).append(c)
    for lst in by_sub.values():
        lst.sort(key=lambda c: (not ABSOLUTE.search(c.text), c.source_tier))  # stable sort: best sources first

    targets: list[Claim] = []
    while len(targets) < n and any(by_sub.values()):
        for sid in list(by_sub):
            if by_sub[sid] and len(targets) < n:
                targets.append(by_sub[sid].pop(0))
    return targets


def _exhausted(state: dict) -> bool:
    return state["calls"] >= MAX_TOOL_CALLS or time.time() - state["start"] > MAX_SECONDS


def _call(state: dict, tracer, name: str, *args):
    if _exhausted(state):
        return None
    state["calls"] += 1
    return call_tool(name, *args, tracer=tracer)


def _challenge(claim: Claim, visited: set[str], state: dict, tracer, limitations: list[str]) -> bool:
    """Try to falsify one claim. Returns True only if the claim was actually tested."""
    try:
        query = llm_json(QUERY_SYSTEM, f"Claim: {claim.text}", CounterQuery).query.strip()
    except Exception as e:  # noqa: BLE001
        tracer.log("skeptic_failed", claim=claim.id, stage="query", error=str(e))
        limitations.append(f"Skeptic could not generate a counter-query for {claim.id}; left untested.")
        return False
    if not query:
        return False

    s = _call(state, tracer, "web_search", query)
    if s is None:
        return False
    if not s.ok:
        limitations.append(f"Skeptic search failed for {claim.id} ({s.error}); left untested.")
        return False

    urls = select_urls(s.data, f"{claim.text} {query}", visited, n=1)
    if not urls:
        limitations.append(f"Skeptic found no usable counter-source for {claim.id}; left untested.")
        return False
    url = urls[0]
    visited.add(url)

    p = _call(state, tracer, "fetch_page", url)
    if p is None:
        return False
    if not p.ok:
        limitations.append(f"Skeptic could not read {url} ({p.error}); {claim.id} left untested.")
        return False

    final_url = p.data.get("url", url)
    text, flagged = sanitize_text(p.data["text"])
    if flagged:
        tracer.log("page_flagged", url=final_url, reason="prompt_injection_pattern", stage="skeptic")

    # Same quarantine as the executor: page text -> extractor -> verified claims only.
    try:
        raw = extract_claims(
            f"Evidence that limits, contradicts or updates this claim: {claim.text}", final_url, text
        )
        counter, _ = verify_claims(
            raw, text, final_url, claim.sub_question_id, claim.text, flagged, lambda: "x", set()
        )
    except Exception as e:  # noqa: BLE001
        tracer.log("skeptic_failed", claim=claim.id, stage="extract", error=str(e))
        limitations.append(f"Skeptic could not analyse {final_url}; {claim.id} left untested.")
        return False

    if not counter:
        claim.verdict = "survived"
        claim.verdict_note = (
            f"No counter-evidence found at {final_url} (searched: '{query}'). "
            "This is a single-source check, not proof."
        )
    else:
        evidence = "\n".join(
            f'- (tier {c.source_tier}) {c.text} | quote: "{c.quote}"' for c in counter
        )
        try:
            j = llm_json(
                JUDGE_SYSTEM,
                f"CLAIM: {claim.text}\n\nCOUNTER-EVIDENCE from {final_url}:\n{evidence}",
                SkepticJudgement,
            )
        except Exception as e:  # noqa: BLE001
            tracer.log("skeptic_failed", claim=claim.id, stage="judge", error=str(e))
            limitations.append(f"Skeptic could not judge {claim.id}; left untested.")
            return False
        claim.verdict = j.verdict
        if claim.verdict == "contradicted" and min(c.source_tier for c in counter) > 2:
            claim.verdict = "weakened"   # forums/blogs alone cannot overturn a claim
            j.note += " (Downgraded: counter-evidence came only from low-tier sources.)"
        claim.verdict_note = f"{j.note.strip()} (source: {final_url})"

    if claim.stale and claim.verdict == "survived":
        claim.verdict = "weakened"
        claim.verdict_note += " The original source is older than 12 months."

    tracer.log("skeptic_verdict", claim=claim.id, verdict=claim.verdict, query=query,
               note=claim.verdict_note[:200])
    return True


def challenge_claims(claims: list[Claim], visited: set[str], tracer,
                     max_claims: int = MAX_CLAIMS_TO_CHALLENGE) -> tuple[list[str], int]:
    """Adversarially test the top claims. Mutates claim.verdict. Returns (limitations, tool_calls)."""
    state = {"calls": 0, "start": time.time()}
    visited = set(visited) | {c.source_url for c in claims}
    limitations: list[str] = []

    web_total = sum(1 for c in claims if c.verified and not _is_primary(c))
    tested = 0
    for c in pick_targets(claims, max_claims):
        if _exhausted(state):
            limitations.append("Skeptic stopped early (its own tool/time budget was reached).")
            break
        if _challenge(c, visited, state, tracer, limitations):
            tested += 1

    if tested < web_total:
        limitations.append(
            f"The Skeptic adversarially checked {tested} of {web_total} web-sourced claims "
            "(cost control); the rest were not challenged."
        )
    tracer.log("skeptic_done", tested=tested, web_claims=web_total, tool_calls=state["calls"])
    return limitations, state["calls"]