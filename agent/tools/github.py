import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def github_stats(repo: str, timeout: int = 10) -> dict:
    """Structured facts about a GitHub repo (primary-source data, not opinions)."""
    if not REPO_RE.match(repo):
        raise ValueError(f"Invalid repo format (expected owner/repo): {repo}")

    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = httpx.get(
        f"https://api.github.com/repos/{repo}", headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    d = resp.json()

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