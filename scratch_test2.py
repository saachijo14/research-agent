from agent import cache, faults
from agent.tools.registry import call_tool

cache.set_mode("auto")

print("--- search")
r = call_tool("web_search", "open source vector database comparison")
print(r.ok, r.error, r.latency_ms, "ms")
url = "https://qdrant.tech"
if r.ok:
    for x in r.data[:3]:
        print(" ", x["title"], "|", x["url"])
    url = r.data[0]["url"]

print("--- same search again (should be much faster: cached)")
r2 = call_tool("web_search", "open source vector database comparison")
print(r2.ok, r2.latency_ms, "ms")

print("--- fetch")
page = call_tool("fetch_page", url)
print(page.ok, page.error, len(page.data["text"]) if page.ok else None, "chars")

print("--- github")
g = call_tool("github_stats", "qdrant/qdrant")
print(g.ok, g.data if g.ok else g.error)

print("--- bad URL (must not crash)")
bad = call_tool("fetch_page", "https://this-domain-does-not-exist-xyz.com", retries=1)
print(bad.ok, bad.error)

print("--- fault: injection")
faults.set_fault("injection")
inj = call_tool("fetch_page", url)
print("poison present:", "IGNORE ALL PREVIOUS" in inj.data["text"])

print("--- fault: timeout")
faults.set_fault("timeout")
t = call_tool("web_search", "anything", retries=1)
print(t.ok, t.error)

print("--- fault: malformed")
faults.set_fault("malformed")
m = call_tool("github_stats", "qdrant/qdrant")
print(m.ok, m.error)

print("--- fault: empty")
faults.set_fault("empty")
e = call_tool("web_search", "open source vector database comparison")
print(e.ok, e.error)

faults.set_fault("none")