import re
from datetime import date

from agent.llm import llm_json
from agent.schemas import Plan, SubQuestion

MAX_SUBQUESTIONS = 5
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

SYSTEM = f"""You are the planning component of a research agent. Today's date is {date.today().isoformat()}.
Break the user's research question into 3 to 5 sub-questions that can each be answered by
searching the web or by looking up a GitHub repository.

Rules:
- Every sub-question must be self-contained: name the specific entities (products, companies, topics) in the text,
  because each one is searched on its own.
- preferred_tool = "github_stats" ONLY for a sub-question about one specific open-source project's popularity,
  license or maintenance activity. Then set github_repo to "owner/repo" (only if you are confident it exists).
  For anything else use "web_search" and leave github_repo null.
- For comparisons of several open-source projects: one github_stats sub-question per project, plus
  one or two web_search sub-questions on qualitative differences (features, tradeoffs, fit for the use case).
- done_criteria: one sentence describing what evidence is enough to answer the question.
- Do NOT answer the question. Only plan."""


def _fallback_plan(question: str) -> Plan:
    return Plan(
        sub_questions=[SubQuestion(id="q1", question=question)],
        done_criteria="At least two independent sources address the question.",
    )


def _normalize(plan: Plan, question: str) -> Plan:
    subs = plan.sub_questions[:MAX_SUBQUESTIONS]
    if not subs:
        return _fallback_plan(question)

    fixed = []
    for i, sq in enumerate(subs, start=1):
        repo = (sq.github_repo or "").strip()
        tool = sq.preferred_tool
        if tool == "github_stats" and not REPO_RE.match(repo):
            tool, repo = "web_search", ""  # bad repo string: don't trust it
        if tool == "web_search":
            repo = ""
        fixed.append(
            SubQuestion(
                id=f"q{i}",
                question=sq.question.strip(),
                preferred_tool=tool,
                github_repo=repo or None,
            )
        )
    return Plan(sub_questions=fixed, done_criteria=plan.done_criteria)


def make_plan(question: str, tracer) -> Plan:
    try:
        plan = llm_json(SYSTEM, f"Research question: {question}", Plan)
    except Exception as e:  # noqa: BLE001
        tracer.log("plan_failed", error=str(e))
        plan = _fallback_plan(question)

    plan = _normalize(plan, question)
    tracer.log("plan", plan=plan.model_dump())
    return plan