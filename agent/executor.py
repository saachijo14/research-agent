import itertools
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from agent.extractor import extract_claims
from agent.schemas import Claim, Plan, SubQuestion
from agent.tools.registry import call_tool
from agent.verifier import (
    claims_from_github,
    sanitize_text,
    select_urls,
    verify_claims,
)

URLS_PER_SUBQUESTION = 2


@dataclass
class Budget:
    """Hard limits so a run can never loop or spend without bound."""
    max_tool_calls: int = 24
    max_seconds: int = 240
    tool_calls: int = 0
    started: float = field(default_factory=time.time)

    def exhausted_reason(self) -> Optional[str]:
        if self.tool_calls >= self.max_tool_calls:
            return f"tool-call budget reached ({self.max_tool_calls})"
        if time.time() - self.started > self.max_seconds:
            return f"time budget reached ({self.max_seconds}s)"
        return None


@dataclass
class ExecResult:
    budget: Budget
    claims: list[Claim] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    pages: dict[str, str] = field(default_factory=dict)      # url -> sanitized text
    visited: set[str] = field(default_factory=set)
    seen_text: set[str] = field(default_factory=set)
    _counter: Iterator[int] = field(default_factory=lambda: itertools.count(1), repr=False)

    def next_id(self) -> str:
        return f"c{next(self._counter)}"


def _call(budget: Budget, tracer, name: str, *args, **kwargs):
    """Spend one unit of budget, or return None if the budget is gone."""
    if budget.exhausted_reason():
        return None
    budget.tool_calls += 1
    return call_tool(name, *args, tracer=tracer, **kwargs)


def _research(sq: SubQuestion, question: str, res: ExecResult, tracer) -> None:
    budget = res.budget
    hints = f"{question} {sq.question}"
    before = len(res.claims)

    # 1) Primary-source tool first, when the plan asked for it.
    if sq.preferred_tool == "github_stats" and sq.github_repo:
        r = _call(budget, tracer, "github_stats", sq.github_repo)
        if r and r.ok:
            res.claims.extend(claims_from_github(r.data, sq.id, res.next_id))
            tracer.log("sub_question_done", id=sq.id, via="github_stats",
                       new_claims=len(res.claims) - before)
            return  # primary data beats blog opinions; also saves budget
        res.limitations.append(
            f"GitHub lookup failed for '{sq.github_repo}' "
            f"({r.error if r else 'budget exhausted'}); fell back to web search for {sq.id}."
        )

    # 2) Web search + page fetch.
    s = _call(budget, tracer, "web_search", sq.question)
    if not s or not s.ok:
        res.limitations.append(
            f"Search failed for {sq.id} ({s.error if s else 'budget exhausted'}); no evidence gathered."
        )
        return

    urls = select_urls(s.data, hints, res.visited, n=URLS_PER_SUBQUESTION)
    if not urls:
        res.limitations.append(f"No usable (non-duplicate, non-blocked) sources found for {sq.id}.")
        return

    for url in urls:
        res.visited.add(url)
        p = _call(budget, tracer, "fetch_page", url)
        if not p or not p.ok:
            res.limitations.append(f"Could not read {url} ({p.error if p else 'budget exhausted'}).")
            continue

        final_url = p.data.get("url", url)
        text, flagged = sanitize_text(p.data["text"])
        if flagged:
            tracer.log("page_flagged", url=final_url, reason="prompt_injection_pattern")
            res.limitations.append(
                f"Source {final_url} contained instruction-like text (possible prompt injection); "
                f"that text was removed and the source was down-weighted."
            )
        if len(text.strip()) < 100:
            continue
        res.pages[final_url] = text

        # Quarantined extraction: page text never reaches any other LLM call.
        try:
            raw = extract_claims(sq.question, final_url, text)
        except Exception as e:  # noqa: BLE001
            tracer.log("extract_failed", url=final_url, error=str(e))
            res.limitations.append(f"Could not extract claims from {final_url}.")
            continue

        accepted, dropped = verify_claims(
            raw, text, final_url, sq.id, hints, flagged, res.next_id, res.seen_text
        )
        for claim_text, why in dropped:
            tracer.log("claim_dropped", reason=why, text=claim_text[:120], url=final_url)
        res.claims.extend(accepted)

    new = len(res.claims) - before
    tracer.log("sub_question_done", id=sq.id, via="web_search", new_claims=new)
    if new == 0:
        res.limitations.append(f"No verified claims were found for {sq.id}: \"{sq.question}\"")


def run_plan(question: str, plan: Plan, tracer, budget: Optional[Budget] = None) -> ExecResult:
    res = ExecResult(budget=budget or Budget())

    for idx, sq in enumerate(plan.sub_questions):
        reason = res.budget.exhausted_reason()
        if reason:
            skipped = [s.id for s in plan.sub_questions[idx:]]
            res.limitations.append(
                f"Research stopped early ({reason}); not researched: {', '.join(skipped)}."
            )
            tracer.log("budget_exhausted", reason=reason, skipped=skipped)
            break
        _research(sq, question, res, tracer)

    tracer.log(
        "exec_done",
        claims=len(res.claims),
        tool_calls=res.budget.tool_calls,
        limitations=len(res.limitations),
    )
    return res