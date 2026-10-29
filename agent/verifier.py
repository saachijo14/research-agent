import re
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from urllib.parse import urlparse

from rapidfuzz import fuzz

from agent.schemas import Claim, RawClaim

STALE_DAYS = 365

# ---------------- prompt-injection detection ----------------
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules)",
    r"disregard\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above|earlier)",
    r"(?:reveal|print|show|repeat)\s+(?:your|the)\s+(?:system\s+prompt|instructions|prompt)",
    r"system\s+prompt",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:\w+\s+){0,2}(?:assistant|bot|agent|ai|model)",
    r"new\s+instructions\s*:",
    r"do\s+not\s+tell\s+the\s+user",
]
_INJ = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


def looks_injected(text: str) -> bool:
    return bool(_INJ.search(text or ""))


def sanitize_text(text: str) -> tuple[str, bool]:
    """Remove instruction-like sentences from page text. Returns (clean_text, was_flagged)."""
    flagged = False
    out_lines = []
    for line in text.splitlines():
        if not _INJ.search(line):
            out_lines.append(line)
            continue
        flagged = True
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [s for s in sentences if not _INJ.search(s)]
        if kept:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines), flagged


# ---------------- source tiers ----------------
OFFICIAL_DOMAINS = {"github.com", "arxiv.org", "pypi.org", "npmjs.com", "huggingface.co"}
REPUTABLE_DOMAINS = {
    "wikipedia.org", "techcrunch.com", "thenewstack.io", "infoworld.com", "venturebeat.com",
    "zdnet.com", "arstechnica.com", "wired.com", "forbes.com", "yourstory.com", "inc42.com",
    "entrackr.com", "economictimes.indiatimes.com", "crunchbase.com", "ibm.com",
    "aws.amazon.com", "cloud.google.com", "microsoft.com", "nvidia.com", "oracle.com",
    "redhat.com", "martinfowler.com",
}
# Low-signal / hard-to-verify sites we never fetch
BLOCKED_DOMAINS = {
    "pinterest.com", "quora.com", "facebook.com", "instagram.com", "tiktok.com",
    "youtube.com", "youtu.be", "linkedin.com", "x.com", "twitter.com", "reddit.com",
}


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _matches(domain: str, group: set[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in group)


def _label(domain: str) -> str:
    parts = domain.split(".")
    return parts[-2] if len(parts) >= 2 else domain


def source_tier(url: str, hints: str = "") -> int:
    """1 = official/primary, 2 = reputable secondary, 3 = weak/unknown.

    Heuristic: a domain whose main label appears in the question (e.g. 'qdrant' in
    qdrant.tech) is treated as the official site. Tune these lists as you evaluate.
    """
    d = _domain(url)
    if (
        _matches(d, OFFICIAL_DOMAINS)
        or d.startswith(("docs.", "developer.", "developers."))
        or d.endswith((".gov", ".edu", ".gov.in", ".ac.in"))
    ):
        return 1
    label = _label(d)
    if len(label) >= 4 and label in hints.lower():
        return 1
    if _matches(d, REPUTABLE_DOMAINS):
        return 2
    return 3


def select_urls(results: list[dict], hints: str, visited: set[str], n: int = 2) -> list[str]:
    """Filter noisy search results: drop blocked/duplicate domains and PDFs, prefer higher tiers."""
    seen_domains: set[str] = set()
    cands = []
    for i, r in enumerate(results):
        url = r.get("url", "")
        d = _domain(url)
        if not d or _matches(d, BLOCKED_DOMAINS):
            continue
        if url in visited or d in seen_domains or url.lower().endswith(".pdf"):
            continue
        seen_domains.add(d)
        cands.append((source_tier(url, hints), i, url))
    cands.sort()
    return [u for _, _, u in cands[:n]]


# ---------------- quote verification ----------------
def _norm(s: str) -> str:
    s = (
        s.lower()
        .replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    return re.sub(r"\s+", " ", s).strip()


def quote_in_text(quote: str, text: str, threshold: int = 90) -> bool:
    q, t = _norm(quote), _norm(text)
    if len(q) < 20:  # too short to be meaningful evidence
        return False
    if q in t:
        return True
    return fuzz.partial_ratio(q, t) >= threshold


# ---------------- dates ----------------
_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%B %Y"]


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"(19|20)\d{2}", s):
        return date(int(s), 12, 31)  # year only: assume end of year (avoid false 'stale')
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ---------------- claim verification ----------------
def verify_claims(
    raw: list[RawClaim],
    page_text: str,
    url: str,
    sub_id: str,
    hints: str,
    flagged: bool,
    next_id: Callable[[], str],
    seen: set[str],
) -> tuple[list[Claim], list[tuple[str, str]]]:
    """Keep only claims whose quote really appears in the page. Returns (accepted, dropped)."""
    accepted: list[Claim] = []
    dropped: list[tuple[str, str]] = []
    tier = 3 if flagged else source_tier(url, hints)

    for rc in raw:
        text = rc.text.strip()
        if not text:
            continue
        if looks_injected(rc.text) or looks_injected(rc.quote):
            dropped.append((text, "injection_pattern_in_claim"))
            continue
        if not quote_in_text(rc.quote, page_text):
            dropped.append((text, "quote_not_found_in_source"))
            continue
        key = _norm(text)
        if key in seen:
            dropped.append((text, "duplicate"))
            continue
        seen.add(key)

        dt = parse_date(rc.published)
        stale = bool(dt and dt < date.today() - timedelta(days=STALE_DAYS))

        accepted.append(
            Claim(
                id=next_id(),
                sub_question_id=sub_id,
                text=text,
                source_url=url,
                quote=rc.quote.strip(),
                source_tier=tier,
                published=rc.published,
                stale=stale,
                verified=True,
            )
        )
    return accepted, dropped


def claims_from_github(data: dict, sub_id: str, next_id: Callable[[], str]) -> list[Claim]:
    """GitHub API data is primary-source: tier 1 and verified by construction."""
    repo, url = data["repo"], data["url"]
    today = date.today().isoformat()
    facts = [
        (
            f"{repo} has {data['stars']:,} GitHub stars and {data['forks']:,} forks.",
            f"GitHub API: stargazers_count={data['stars']}, forks_count={data['forks']}",
        ),
        (
            (
                f"{repo} is released under the {data['license']} license."
                if data.get("license") not in (None, "NOASSERTION")
                else f"{repo}: GitHub could not auto-detect a standard license; check the repository's LICENSE file."
            ),
            f"GitHub API: license={data.get('license')}",
        ),
        (
            f"{repo} was last pushed to on {str(data['last_push'])[:10]}, has "
            f"{data['open_issues']:,} open issues, and archived={data['archived']}.",
            f"GitHub API: pushed_at={data['last_push']}, open_issues_count={data['open_issues']}, "
            f"archived={data['archived']}",
        ),
    ]
    return [
        Claim(
            id=next_id(),
            sub_question_id=sub_id,
            text=text,
            source_url=url,
            quote=quote,
            source_tier=1,
            published=today,
            verified=True,
        )
        for text, quote in facts
    ]