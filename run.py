import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape as esc
from rich.panel import Panel
from rich.table import Table

from agent import cache, faults
from agent.pipeline import RunOutput, run_agent, run_baseline

console = Console()
COLORS = {"Low": "red", "Medium": "yellow", "High": "green"}


def render(out: RunOutput, show_ledger: bool, show_trace: bool) -> None:
    a = out.answer
    console.print(Panel(esc(a.question), title="Question"))
    console.print(Panel(esc(a.short_answer), title="Short answer"))

    console.print("\n[bold]Key findings[/bold]")
    if a.key_findings:
        for i, f in enumerate(a.key_findings, 1):
            console.print(f"  {i}. {esc(f)}")
    else:
        console.print("  (none - insufficient verified evidence)")

    console.print("\n[bold]Sources used[/bold]")
    for s in a.sources or ["(none)"]:
        console.print(f"  - {esc(s)}")

    color = COLORS[a.confidence]
    console.print(f"\n[bold]Confidence:[/bold] [{color}]{a.confidence}[/{color}] (score {a.confidence_score})")

    for title, items in (
        ("Limitations", a.limitations),
        ("Assumptions", a.assumptions),
        ("Suggested next steps", a.next_steps),
    ):
        console.print(f"\n[bold]{title}[/bold]")
        for it in items or ["(none)"]:
            console.print(f"  - {esc(it)}")

    s = out.stats
    console.print(
        f"\n[dim]latency {s['latency_s']}s | tool calls {s['tool_calls']} | llm calls {s['llm_calls']} | "
        f"tokens {s['prompt_tokens']}+{s['completion_tokens']} | est. cost ${s['est_cost_usd']}[/dim]"
    )
    if s["confidence_breakdown"]:
        console.print(f"[dim]confidence breakdown: {esc(str(s['confidence_breakdown']))}[/dim]")
    console.print(f"[dim]trace: {esc(str(out.tracer.path))}[/dim]")

    if show_ledger and out.claims:
        t = Table(title="Evidence ledger", show_lines=True)
        for col in ("ID", "Tier", "Verdict", "Claim", "Source"):
            t.add_column(col, overflow="fold")
        for c in out.claims:
            verdict = c.verdict + (" (stale)" if c.stale else "")
            t.add_row(esc(c.id), str(c.source_tier), esc(verdict), esc(c.text), esc(c.source_url))
        console.print(t)

    if show_trace:
        t = Table(title="Trace", show_lines=False)
        t.add_column("+s")
        t.add_column("event")
        t.add_column("details", overflow="fold")
        t0 = out.tracer.events[0]["t"] if out.tracer.events else 0
        for ev in out.tracer.events:
            rest = {k: v for k, v in ev.items() if k not in ("t", "event")}
            if ev["event"] == "tool_call":
                detail = (f"{rest['tool']} ok={rest['ok']} {rest['latency_ms']}ms "
                          f"cached={rest['cached']} {rest.get('error') or ''}")
            else:
                detail = json.dumps(rest, default=str)[:160]
            t.add_row(f"{ev['t'] - t0:.1f}", esc(ev["event"]), esc(detail))
        console.print(t)


def main() -> None:
    p = argparse.ArgumentParser(description="Research agent with an evidence ledger and a Skeptic.")
    p.add_argument("question")
    p.add_argument("--mode", choices=["agent", "baseline"], default="agent")
    p.add_argument("--fault", choices=sorted(faults.VALID_FAULTS), default="none",
                   help="inject a failure to test robustness")
    p.add_argument("--replay", action="store_true", help="use cached tool results only (no network)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-skeptic", action="store_true")
    p.add_argument("--ledger", action="store_true", help="show the evidence ledger")
    p.add_argument("--trace", action="store_true", help="show the trace of this run")
    p.add_argument("--json", action="store_true", help="print JSON instead of the formatted view")
    p.add_argument("--out", help="save the full result as JSON to this path")
    args = p.parse_args()

    faults.set_fault(args.fault)
    if args.replay:
        cache.set_mode("replay")
    elif args.no_cache:
        cache.set_mode("off")

    if args.mode == "baseline":
        out = run_baseline(args.question)
    else:
        out = run_agent(args.question, use_skeptic=not args.no_skeptic)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out.to_dict(), indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out.to_dict(), indent=2))
    else:
        render(out, args.ledger, args.trace)


if __name__ == "__main__":
    main()