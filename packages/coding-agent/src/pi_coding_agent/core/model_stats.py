"""Model invocation shares and estimated costs from persisted session messages."""
from collections import Counter
from math import isfinite

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
    costs = Counter()
    priced = Counter()
    unclassified = Counter()
    unrecorded_auxiliary_failures = 0

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
        if (decision or {}).get('usage_complete') is False:
            known = False
        if message.get('stop_reason', message.get('stopReason')) in ('error', 'aborted') and not any(fields.values()):
            known = False
        missing_usage += not known
        # Tau normalizes uncached input and cached input into separate fields.
        # Do not add the reported total a second time.
        classified = sum(fields.values())
        reported = int(usage.get('total_tokens', usage.get('totalTokens', 0)) or 0)
        total = max(classified, reported)
        residual = max(0, reported - classified)
        unclassified[key] += residual
        matches = decision and decision.get('provider') == provider and decision.get('model') == model
        rates = (decision.get('pricing') or {}) if matches else {}
        valid_rates = all(k in rates and isinstance(rates[k], (int, float))
                          and isfinite(rates[k]) and rates[k] >= 0
                          for k, count in fields.items() if count)
        if known and not residual and valid_rates and rates:
            costs[key] += sum(count * rates[k] for k, count in fields.items() if count) / 1_000_000
            priced[key] += 1
        tokens[key] += total
        breakdown.setdefault(key, Counter()).update(fields)

    for entry in entries:
        if hasattr(entry, "data"):
            entry = {**entry.data, "type": entry.type}
        if entry.get("customType") == "tau.router_selection":
            selection = entry.get("data", {})
            continue
        custom = entry.get('customType', '')
        purpose = ('intention' if custom.startswith('tau.intention.') else
                   'reviewer' if custom.startswith('tau.turn_review.') else None)
        if purpose and custom.endswith('.invocation'):
            data = entry.get('data', {})
            add(data.get('message') or {}, {**data, 'level': data.get('level') or 'max'}, purpose)
            continue
        if purpose and custom.endswith(('.completed', '.failed')):
            data = entry.get('data', {})
            if data.get('usage_recording') == 'per_invocation':
                continue
            legacy = [m for m in data.get('messages', []) if m.get('role') == 'assistant']
            for message in legacy:
                add(message, {**data, 'level': data.get('level') or 'max'}, purpose)
            if custom.endswith('.failed') and not legacy:
                unrecorded_auxiliary_failures += 1
            continue
        message = entry.get("message", {})
        if entry.get("type") != "message" or message.get("role") != "assistant":
            continue
        decision, selection = selection, None
        add(message, decision, 'worker')
    total = sum(counts.values())
    total_tokens = sum(tokens.values())
    return {"total": total, "total_tokens": total_tokens, "missing_usage": missing_usage,
            "incomplete": incomplete, "unrecorded_auxiliary_failures": unrecorded_auxiliary_failures,
            "estimated_cost": sum(costs.values()), "priced_calls": sum(priced.values()), "models": [
        {"provider": key[0], "model": key[1], "tier": key[2], "effort": key[3],
         "purpose": key[4], "count": counts[key], "tokens": tokens[key],
         "usage": dict(breakdown[key]), "unclassified_tokens": unclassified[key],
         "estimated_cost": costs[key] if priced[key] else None, "priced_calls": priced[key],
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
        if row['unclassified_tokens']:
            lines.append(f"  unclassified tokens {row['unclassified_tokens']:,} (included in total)")
        if row['estimated_cost'] is not None:
            lines.append(f"  configured-rate estimate ${row['estimated_cost']:.6f} · {row['priced_calls']}/{row['count']} calls priced")
    lines.append("Includes worker, intention and reviewer tokens (including cache); not cost shares.")
    if stats['priced_calls']:
        lines.append(f"Configured-rate estimate ${stats['estimated_cost']:.6f} · {stats['priced_calls']}/{stats['total']} calls priced; not billed cost or routing savings.")
    if stats['priced_calls'] < stats['total']:
        lines.append("Unpriced calls have missing/incomplete rate snapshots or usage; not assumed free.")
    if stats['unrecorded_auxiliary_failures']:
        lines.append(f"{stats['unrecorded_auxiliary_failures']} historical failed auxiliary runs lack call usage.")
    if stats['missing_usage']:
        lines.append(f"{stats['missing_usage']} calls missing token usage; percentages cover recorded tokens only.")
    if stats["incomplete"]:
        lines.append(f"Failed/aborted calls included: {stats['incomplete']}")
    return "\n".join(lines)
