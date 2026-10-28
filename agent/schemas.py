from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- Planning ----------
class SubQuestion(BaseModel):
    id: str = Field(description="Short id like q1, q2")
    question: str
    preferred_tool: Literal["web_search", "github_stats"] = "web_search"
    github_repo: Optional[str] = Field(
        default=None,
        description="owner/repo, only if preferred_tool is github_stats",
    )


class Plan(BaseModel):
    sub_questions: list[SubQuestion]
    done_criteria: str


# ---------- Extraction ----------
class RawClaim(BaseModel):
    """What the quarantined extractor returns (no ids, no verification)."""
    text: str
    quote: str = Field(description="Exact sentence copied from the page")
    published: Optional[str] = None


class ExtractionResult(BaseModel):
    claims: list[RawClaim]


# ---------- Ledger ----------
class Claim(BaseModel):
    id: str
    sub_question_id: str
    text: str
    source_url: str
    quote: str
    source_tier: Literal[1, 2, 3] = 3   # 1 official, 2 reputable, 3 weak
    published: Optional[str] = None
    stale: bool = False
    verified: bool = False
    verdict: Literal["untested", "survived", "weakened", "contradicted"] = "untested"
    verdict_note: str = ""


# ---------- Skeptic ----------
class CounterQuery(BaseModel):
    query: str


class SkepticJudgement(BaseModel):
    verdict: Literal["survived", "weakened", "contradicted"]
    note: str


# ---------- Final output ----------
class FinalAnswer(BaseModel):
    question: str
    short_answer: str
    key_findings: list[str]
    sources: list[str]
    confidence: Literal["Low", "Medium", "High"]
    confidence_score: float
    limitations: list[str]
    assumptions: list[str]
    next_steps: list[str]