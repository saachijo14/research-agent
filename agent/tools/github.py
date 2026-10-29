import os
import re
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _search_repo(name: str, timeout: int) -> Optional[str]:
    """The planner guessed an owner that does not exist: find the real repo by exact name."""
    r = httpx.get(
        "https://api.github.com/search/repositories",
        params={"q": f"{name} in:name", "sort": "stars", "order": "desc", "per_page": 5},
        headers=_headers(),
        timeout=timeout,
    )
    if r.status_code != 200:
        return None
    for item in r.json().get("items", []):
        if (
            item["name"].lower() == name.lower()
            and item["stargazers_count"] >= 1000
            and not item.get("fork")
        ):
            return item["full_name"]
    return None


def github_stats(repo: str, timeout: int = 10, _resolved: bool = False) -> dict:
    """Structured facts about a GitHub repo (primary-source data, not opinions)."""
    if not REPO_RE.match(repo):
        raise ValueError(f"Invalid repo format (expected owner/repo): {repo}")

    resp = httpx.get(
        f"https://api.github.com/repos/{repo}",
        headers=_headers(),
        timeout=timeout,
        follow_redirects=True,
    )

    if resp.status_code == 404 and not _resolved:
        found = _search_repo(repo.split("/")[1], timeout)
        if found:
            return github_stats(found, timeout=timeout, _resolved=True)

    resp.raise_for_status()
    d = resp.json()

    # If this is a fork, report the original (upstream) project instead
    if d.get("fork") and (d.get("source") or {}).get("full_name"):
        return github_stats(d["source"]["full_name"], timeout=timeout, _resolved=True)

    return {
        "repo": d["full_name"],
        "url": d["html_url"],
        "description": d.get("description"),
        "stars": d["stargazers_count"],
        "forks": d["forks_count"],
        "open_issues": d["open_issues_count"],
        "license": (d.get("license") or {}).get("spdx_id"),
        "last_push": d["pushed_at"],
        "archived": d["archived"],
    }