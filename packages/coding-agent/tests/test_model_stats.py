from pi_coding_agent.core.model_stats import model_stats, render_model_stats


def test_old_sessions_providers_errors_and_non_message_entries():
    def message(provider, model, stop="stop"):
        return {"type": "message", "message": {"role": "assistant", "provider": provider,
                                                  "model": model, "stopReason": stop}}
    entries = [message("a", "same"), message("a", "same", "toolUse"), message("b", "same"),
               message("a", "same", "error"), message("a", "same", "aborted"),
               {"type": "custom", "customType": "tau.router_selection", "data": {"model": "other"}},
               {"type": "message", "message": {"role": "user", "content": "hello"}}]
    stats = model_stats(entries)
    assert stats["total"] == 5 and stats["incomplete"] == 2
    assert [r["count"] for r in stats["models"]] == [4, 1]
    output = render_model_stats(stats)
    assert stats['missing_usage'] == 5
    assert all(r['percent'] is None for r in stats['models'])
    assert "a/same" in output and "b/same" in output
    assert "Failed/aborted calls included: 2" in output


def test_empty_session_chart():
    assert "No invocations recorded yet." in render_model_stats(model_stats([]))


def test_luna_tiers_split_and_decisions_are_consumed_once():
    def decision(tier, effort, model="luna"):
        return {"type": "custom", "customType": "tau.router_selection", "data": {
            "provider": "openai", "model": model, "level": tier, "reasoning": effort}}
    def response(stop="stop"):
        return {"type": "message", "message": {"role": "assistant", "provider": "openai",
                "model": "luna", "stop_reason": stop}}
    rows = [decision("ultra-light", "low"), response(), decision("light", "high"), response(),
            decision("light", "high"), response("error"), response(),
            decision("max", "medium", "astra"), response()]
    stats = model_stats(rows)
    assert stats["total"] == 5 and stats["incomplete"] == 1
    assert {r["tier"]: r["count"] for r in stats["models"]} == {
        "ultra-light": 1, "light": 2, "unrecorded": 2}
    chart = render_model_stats(stats)
    assert "ultra-light · low" in chart and "light · high" in chart
    assert "tier/effort not recorded" in chart


def test_token_shares_include_auxiliary_calls_and_cache_without_double_counting():
    def response(model, usage):
        return {'role': 'assistant', 'provider': 'test', 'model': model, 'usage': usage}
    entries = [
        {'type': 'message', 'message': response('small', {'input': 10, 'output': 10})},
        {'type': 'message', 'message': response('small', {'totalTokens': 20})},
        {'type': 'custom', 'customType': 'tau.intention.completed', 'data': {
            'reasoning': 'high', 'messages': [response('configured-max', {
                'input': 20, 'cache_read': 100, 'output': 40, 'total_tokens': 160})]}},
        {'type': 'custom', 'customType': 'tau.turn_review.completed', 'data': {
            'reasoning': 'high', 'messages': [response('configured-max', {
                'inputTokens': 100, 'cacheRead': 600, 'cacheWrite': 50, 'outputTokens': 50}),
                {'role': 'toolResult', 'content': 'ignored'}]}},
    ]
    stats = model_stats(entries)
    assert stats['total_tokens'] == 1000 and stats['total'] == 4
    assert {r['purpose']: r['percent'] for r in stats['models']} == {
        'worker': 4, 'intention': 16, 'reviewer': 80}
    output = render_model_stats(stats)
    assert '80.0%' in output and '800 tokens' in output
    assert 'reviewer' in output and 'cached input 600' in output


def test_failed_usage_counts_and_unknown_usage_is_disclosed():
    stats = model_stats([
        {'type': 'message', 'message': {'role': 'assistant', 'model': 'm',
         'stop_reason': 'error', 'usage': {'input': 99, 'output': 1}}},
        {'type': 'message', 'message': {'role': 'assistant', 'model': 'm'}},
    ])
    assert stats['total_tokens'] == 100 and stats['missing_usage'] == 1
    assert stats['models'][0]['percent'] == 100
    assert '1 calls missing token usage' in render_model_stats(stats)
