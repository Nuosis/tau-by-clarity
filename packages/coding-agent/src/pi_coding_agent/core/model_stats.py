"""Model invocation shares from persisted session messages, including old sessions."""
from collections import Counter


def model_stats(entries):
    counts = Counter()
    incomplete = 0
    selection = None
    for entry in entries:
        if hasattr(entry, "data"):
            entry = {**entry.data, "type": entry.type}
        if entry.get("customType") == "tau.router_selection":
            selection = entry.get("data", {})
            continue
        message = entry.get("message", {})
        if entry.get("type") != "message" or message.get("role") != "assistant":
            continue
        decision, selection = selection, None
        provider = message.get("provider") or "unrecorded"
        model = message.get("model") or "unrecorded"
        tier, effort = "unrecorded", "unrecorded"
        if decision and decision.get("provider") == provider and decision.get("model") == model:
            tier = decision.get("level") or "unrecorded"
            effort = decision.get("reasoning") or "off"
        if message.get("stop_reason", message.get("stopReason")) in ("error", "aborted"):
            incomplete += 1
            continue
        counts[(provider, model, tier, effort)] += 1
    total = sum(counts.values())
    return {"total": total, "incomplete": incomplete, "models": [
        {"provider": provider, "model": model, "tier": tier, "effort": effort, "count": count, "percent": count * 100 / total}
        for (provider, model, tier, effort), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]}


def render_model_stats(stats):
    lines = ["Model usage · session", f"Share of completed invocations ({stats['total']} total)"]
    if not stats["total"]:
        lines.append("No completed invocations recorded yet.")
    for row in stats["models"]:
        filled = round(row["percent"] * 24 / 100)
        label = (f"{row['tier']} · {row['effort']}" if row['tier'] != "unrecorded"
                 else "tier/effort not recorded")
        lines.extend([f"{row['provider']}/{row['model']} · {label}",
                      f"  {'█' * filled}{'░' * (24 - filled)} {row['percent']:5.1f}%  ({row['count']})"])
    lines.append("Includes tool-use responses; percentages are not token or cost shares.")
    if stats["incomplete"]:
        lines.append(f"Excluded failed/aborted responses: {stats['incomplete']}")
    return "\n".join(lines)
