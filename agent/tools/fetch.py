from urllib.parse import urlparse

import httpx
import trafilatura

MAX_CHARS = 8000  # cost control: never send more than this to the extractor
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/0.1)"}


def fetch_page(url: str, timeout: int = 10) -> dict:
    """Download a page and return clean main-content text, size-capped."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {url}")

    resp = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        raise ValueError(f"Unsupported content-type: {ctype}")

    text = trafilatura.extract(resp.text, include_comments=False, include_tables=True) or ""
    return {"url": str(resp.url), "text": text[:MAX_CHARS]}