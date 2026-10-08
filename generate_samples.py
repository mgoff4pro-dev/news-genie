"""Create sample outputs for the report (technology, finance, sports).

Run after setting your API keys:   python generate_samples.py
Writes sample_outputs.md, which you can paste into your report.
"""

from datetime import datetime

from newsgenie import config, graph

RUNS = [
    ("Technology headlines", "", "headlines", "technology"),
    ("Finance headlines", "", "headlines", "finance"),
    ("Sports headlines", "", "headlines", "sports"),
    ("Topic news request", "What's the latest news about artificial intelligence?", "chat", "technology"),
    ("Web lookup", "What is the current federal funds rate?", "chat", "finance"),
    ("General question", "Explain what a stock market index is in simple terms.", "chat", "general"),
]


def main() -> None:
    thread = graph.new_thread_id()
    status = config.key_status()
    lines = [
        "# NewsGenie sample outputs",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        "Services: " + ", ".join(f"{k.replace('_API_KEY','').replace('_KEY','')}="
                                 f"{'on' if v else 'off'}" for k, v in status.items()),
        "",
    ]
    for title, query, mode, category in RUNS:
        r = graph.run(query, thread, mode=mode, category=category)
        lines += [f"## {title}", ""]
        if query:
            lines += [f"**User:** {query}", ""]
        lines += [f"**Route:** {r.get('intent')} -> {' > '.join(r.get('trace', []))}",
                  f"**Provider:** {r.get('provider') or 'n/a'} · **Time:** {r.get('elapsed')}s", "",
                  r.get("answer", ""), ""]
        for a in r.get("articles", [])[:5]:
            lines.append(f"- [{a['title']}]({a['url']}) ({a.get('source')}, "
                         f"{a.get('credibility_label')} credibility)")
        for e in r.get("errors", []):
            lines.append(f"> Warning: {e}")
        lines.append("")
        print(f"done: {title}")
    with open("sample_outputs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote sample_outputs.md")


if __name__ == "__main__":
    main()
