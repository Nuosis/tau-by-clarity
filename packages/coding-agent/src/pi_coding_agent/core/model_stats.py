"""Model invocation shares from persisted session messages, including old sessions."""
from collections import Counter


def _token_usage(message):
    usage = message.get('usage') or {}
    def value(*keys):
        return next((int(usage[key] or 0) for key in keys if key in usage), 0)
    return {'input': value('input', 'inputTokens'), 'output': value('output', 'outputTokens'),
            'cache_read': value('cache_read', 'cacheRead', 'cacheReadInputTokens'),
            'cache_write': value('cache_write', 'cacheWrite', 'cacheCreationInputTokens')}


def model_stats(entries):
    counts = Counter()
    tokens = Counter()
    breakdown = {}
    missing_usage = 0
    incomplete = 0
    selection = None

    def add(message, decision, purpose):
        nonlocal missing_usage, incomplete
        provider = message.get('provider') or 'unrecorded'
        model = message.get('model') or 'unrecorded'
        tier, effort = 'unrecorded', 'unrecorded'
        if decision and decision.get('provider') == provider and decision.get('model') == model:
            tier = decision.get('level') or 'unrecorded'
            effort = decision.get('reasoning') or 'off'
        key = (provider, model, tier, effort, purpose)
        counts[key] += 1
        incomplete += message.get('stop_reason', message.get('stopReason')) in ('error', 'aborted')
        usage = message.get('usage') or {}
        fields = _token_usage(message)
        known = any(k in usage for k in (
            'input', 'inputTokens', 'output', 'outputTokens', 'cache_read', 'cacheRead',
            'cacheReadInputTokens', 'cache_write', 'cacheWrite', 'cacheCreationInputTokens',
            'total_tokens', 'totalTokens'))
        missing_usage += not known
        # Tau normalizes uncached input and cached input into separate fields.
        # Do not add the reported total a second time.
        total = sum(fields.values())
        if not total:
            total = int(usage.get('total_tokens', usage.get('totalTokens', 0)) or 0)
        tokens[key] += total
        breakdown.setdefault(key, Counter()).update(fields)

    for entry in entries:
        if hasattr(entry, "data"):
            entry = {**entry.data, "type": entry.type}
        if entry.get("customType") == "tau.router_selection":
            selection = entry.get("data", {})
            continue
        purpose = {'tau.intention.completed': 'intention',
                   'tau.turn_review.completed': 'reviewer'}.get(entry.get('customType'))
        if purpose:
            data = entry.get('data', {})
            for message in data.get('messages', []):
                if message.get('role') == 'assistant':
                    add(message, {**data, 'level': 'max'}, purpose)
            continue
        message = entry.get("message", {})
        if entry.get("type") != "message" or message.get("role") != "assistant":
            continue
        decision, selection = selection, None
        add(message, decision, 'worker')
    total = sum(counts.values())
    total_tokens = sum(tokens.values())
    return {"total": total, "total_tokens": total_tokens, "missing_usage": missing_usage,
            "incomplete": incomplete, "models": [
        {"provider": key[0], "model": key[1], "tier": key[2], "effort": key[3],
         "purpose": key[4], "count": counts[key], "tokens": tokens[key],
         "usage": dict(breakdown[key]),
         "percent": tokens[key] * 100 / total_tokens if total_tokens else None}
        for key in sorted(counts, key=lambda key: (-tokens[key], -counts[key], key))
    ]}


def render_model_stats(stats):
    lines = ["Model usage · session",
             f"Share of recorded tokens ({stats['total_tokens']:,} tokens; {stats['total']} calls)"]
    if not stats["total"]:
        lines.append("No invocations recorded yet.")
    for row in stats["models"]:
        filled = round((row["percent"] or 0) * 24 / 100)
        percent = f"{row['percent']:5.1f}%" if row['percent'] is not None else '   n/a'
        label = (f"{row['tier']} · {row['effort']}" if row['tier'] != "unrecorded"
                 else "tier/effort not recorded")
        usage = row['usage']
        lines.extend([f"{row['provider']}/{row['model']} · {label} · {row['purpose']}",
                      f"  {'█' * filled}{'░' * (24 - filled)} {percent}  {row['tokens']:,} tokens · {row['count']} calls",
                      f"  input {usage['input']:,} · cached input {usage['cache_read']:,} · cache write {usage['cache_write']:,} · output {usage['output']:,}"])
    lines.append("Includes worker, intention and reviewer tokens (including cache); not cost shares.")
    if stats['missing_usage']:
        lines.append(f"{stats['missing_usage']} calls missing token usage; percentages cover recorded tokens only.")
    if stats["incomplete"]:
        lines.append(f"Failed/aborted calls included: {stats['incomplete']}")
    return "\n".join(lines)
