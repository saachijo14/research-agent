import os

import httpx
from dotenv import load_dotenv

load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"


def web_search(query: str, timeout: int = 10, max_results: int = 6) -> list[dict]:
    """Search the web via Tavily. Returns [{title, url, snippet}, ...]."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    resp = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": max_results, "search_depth": "basic"},
        timeout=timeout,
    )
    resp.raise_for_status()

    results = resp.json().get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:500],
        }
        for r in results
        if r.get("url")
    ]