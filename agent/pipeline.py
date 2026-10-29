import time
from dataclasses import dataclass
from typing import Optional

from agent import llm
from agent.confidence import compute_confidence
from agent.executor import Budget, run_plan
from agent.planner import make_plan
from agent.schemas import Claim, FinalAnswer
from agent.skeptic import challenge_claims
from agent.synthesizer import baseline_answer, insufficient_answer, synthesize
from agent.trace import Tracer

# USD per 1M tokens. Set these to your model's pricing (defaults are gpt-4o-mini-style numbers).
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60


@dataclass
class RunOutput:
    answer: FinalAnswer
    claims: list[Claim]
    stats: dict
    tracer: Tracer

    def to_dict(self) -> dict:
        return {
            "answer": self.answer.model_dump(),
            "ledger": [c.model_dump() for c in self.claims],
            "stats": self.stats,
            "trace_file": str(self.tracer.path),
        }


def _stats(t0: float, tool_calls: int, breakdown: Optional[dict] = None) -> dict:
    u = llm.usage
    cost = u["prompt_tokens"] / 1e6 * PRICE_IN_PER_M + u["completion_tokens"] / 1e6 * PRICE_OUT_PER_M
    return {
        "latency_s": round(time.time() - t0, 1),
        "tool_calls": tool_calls,
        "llm_calls": u["calls"],
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "est_cost_usd": round(cost, 4),
        "confidence_breakdown": breakdown or {},
    }


def run_agent(question: str, tracer: Optional[Tracer] = None, max_tool_calls: int = 24,
              use_skeptic: bool = True) -> RunOutput:
    """The full pipeline. Never raises: any failure becomes a Low-confidence answer."""
    tracer = tracer or Tracer()
    llm.reset_usage()
    t0 = time.time()
    claims: list[Claim] = []
    tool_calls = 0
    breakdown: dict = {}

    try:
        tracer.log("run_start", question=question, mode="agent", use_skeptic=use_skeptic)
        plan = make_plan(question, tracer)

        res = run_plan(question, plan, tracer, Budget(max_tool_calls=max_tool_calls))
        claims = res.claims
        limitations = list(res.limitations)
        tool_calls = res.budget.tool_calls

        if use_skeptic and claims:
            sk_lims, sk_calls = challenge_claims(claims, res.visited, tracer)
            limitations += sk_lims
            tool_calls += sk_calls

        score, bucket, breakdown = compute_confidence(claims, plan.sub_questions)
        tracer.log("confidence", score=score, bucket=bucket, breakdown=breakdown)

        answer = synthesize(question, claims, limitations, score, bucket)
    except Exception as e:  # noqa: BLE001
        tracer.log("run_failed", error=f"{type(e).__name__}: {e}")
        answer = insufficient_answer(question, [f"The run failed: {type(e).__name__}: {e}"], 0.0, "Low")

    stats = _stats(t0, tool_calls, breakdown)
    tracer.log("run_end", confidence=answer.confidence, score=answer.confidence_score, **{
        k: v for k, v in stats.items() if k != "confidence_breakdown"})
    return RunOutput(answer=answer, claims=claims, stats=stats, tracer=tracer)


def run_baseline(question: str, tracer: Optional[Tracer] = None) -> RunOutput:
    """One LLM call, no tools, same output schema. Used for the agent-vs-baseline comparison."""
    tracer = tracer or Tracer()
    llm.reset_usage()
    t0 = time.time()
    try:
        tracer.log("run_start", question=question, mode="baseline")
        answer = baseline_answer(question)
    except Exception as e:  # noqa: BLE001
        tracer.log("run_failed", error=f"{type(e).__name__}: {e}")
        answer = insufficient_answer(question, [f"The baseline failed: {type(e).__name__}: {e}"], 0.0, "Low")
    stats = _stats(t0, 0)
    tracer.log("run_end", confidence=answer.confidence, **{
        k: v for k, v in stats.items() if k != "confidence_breakdown"})
    return RunOutput(answer=answer, claims=[], stats=stats, tracer=tracer)