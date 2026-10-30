import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import yaml  # noqa: E402

from agent.pipeline import run_baseline  # noqa: E402

URL = re.compile(r"https?://[^\s)\]>,;\"']+")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgentEval/0.1)"}


def resolves(url: str) -> bool:
    try:
        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False


cases = yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))["cases"]
total_listed = total_urls = total_ok = 0

for c in cases:
    if c.get("fault", "none") != "none":
        continue
    out = run_baseline(c["question"])
    if out.stats["prompt_tokens"] == 0:
        print(f"\n[{c['id']}] LLM unreachable, waiting 30s and retrying once ...")
        time.sleep(60)
        out = run_baseline(c["question"])
    if out.stats["prompt_tokens"] == 0:
        print(f"[{c['id']}] INVALID (LLM down), skipped")
        continue
    a = out.answer
    urls = [u.rstrip(".") for s in a.sources for u in URL.findall(s)]
    ok = [u for u in urls if resolves(u)]
    total_listed += len(a.sources)
    total_urls += len(urls)
    total_ok += len(ok)
    print(f"\n[{c['id']}] conf={a.confidence} findings={len(a.key_findings)} "
          f"sources_listed={len(a.sources)} with_url={len(urls)} resolving={len(ok)}")
    for u in urls:
        print("   ", "OK  " if u in ok else "DEAD", u)
    time.sleep(3)

pct = round(100 * total_ok / total_urls) if total_urls else 0
print(f"\nTOTAL: {total_listed} sources listed, {total_urls} had a URL, {total_ok} resolve ({pct}%).")
print("Note: a resolving URL does not prove it supports the claim; a dead URL is a likely fabrication.")