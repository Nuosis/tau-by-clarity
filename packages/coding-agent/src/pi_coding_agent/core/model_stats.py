"""Model invocation shares from persisted session messages, including old sessions."""
from collections import Counter


def model_stats(entries):
    counts = Counter()
    incomplete = 0
    for entry in entries:
        if hasattr(entry, "data"):
            entry = {**entry.data, "type": entry.type}
        message = entry.get("message", {})
        if entry.get("type") != "message" or message.get("role") != "assistant":
            continue
        if message.get("stop_reason", message.get("stopReason")) in ("error", "aborted"):
            incomplete += 1
            continue
        counts[(message.get("provider") or "unrecorded", message.get("model") or "unrecorded")] += 1
    total = sum(counts.values())
    return {"total": total, "incomplete": incomplete, "models": [
        {"provider": provider, "model": model, "count": count, "percent": count * 100 / total}
        for (provider, model), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]}


def render_model_stats(stats):
    lines = ["Model usage · session", f"Share of completed invocations ({stats['total']} total)"]
    if not stats["total"]:
        lines.append("No completed invocations recorded yet.")
    for row in stats["models"]:
        filled = round(row["percent"] * 24 / 100)
        lines.extend([f"{row['provider']}/{row['model']}",
                      f"  {'█' * filled}{'░' * (24 - filled)} {row['percent']:5.1f}%  ({row['count']})"])
    lines.append("Includes tool-use responses; percentages are not token or cost shares.")
    if stats["incomplete"]:
        lines.append(f"Excluded failed/aborted responses: {stats['incomplete']}")
    return "\n".join(lines)
