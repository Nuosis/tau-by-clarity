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
    assert stats["total"] == 3 and stats["incomplete"] == 2
    assert [r["count"] for r in stats["models"]] == [2, 1]
    output = render_model_stats(stats)
    assert "66.7%  (2)" in output and "33.3%  (1)" in output
    assert "a/same" in output and "b/same" in output
    assert "Excluded failed/aborted responses: 2" in output


def test_empty_session_chart():
    assert "No completed invocations recorded yet." in render_model_stats(model_stats([]))


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
    assert stats["total"] == 4 and stats["incomplete"] == 1
    assert {r["tier"]: r["count"] for r in stats["models"]} == {
        "ultra-light": 1, "light": 1, "unrecorded": 2}
    chart = render_model_stats(stats)
    assert "ultra-light · low" in chart and "light · high" in chart
    assert "tier/effort not recorded" in chart
