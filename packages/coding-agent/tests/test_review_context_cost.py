"""Projection/usage functionality, not an LLM effectiveness evaluation."""
import copy
import json
from types import SimpleNamespace

import pytest
from pi_agent.types import AgentContext
from pi_ai.types import AssistantMessage, EventDone, TextContent, Usage
from pi_coding_agent import active_compression
from pi_coding_agent.active_compression.ccr import CCRStore
from pi_coding_agent.core.review_context import build_review_payload
from pi_coding_agent.core.turn_review import Intention, review_turn
from pi_coding_agent.core.model_stats import model_stats
from .test_model_router import make_session, model_command


def test_focused_projection_preserves_source_and_stable_prefix(tmp_path, monkeypatch):
    store = CCRStore(str(tmp_path / 'ccr.db'))
    monkeypatch.setattr(active_compression, '_store', store)
    lines = [f'worker-{i} heartbeat status=200' for i in range(2000)]
    lines[777] = 'audit_marker deployment_http_status=503'
    source = '\n'.join(lines)
    messages = [dict(role='user', timestamp=0, content='Check audit_marker deployment_http_status'),
                dict(role='toolResult', timestamp=0, tool_call_id='log', tool_name='read', is_error=True,
                     content=[dict(type='text', text=source)], details={'raw': source})]
    context = AgentContext(system_prompt='Keep authorization and instructions.', messages=messages, tools=[])
    original = copy.deepcopy(context.messages)
    intent = Intention(outcome='Check audit_marker deployment_http_status',
                       completion_evidence=['Report observed status'], scope='Read only')
    payload, meta = build_review_payload(context, intent)
    packet = json.loads(payload)
    assert len(payload) < len(source) / 5
    assert packet['messages'][1]['message']['is_error'] is True
    assert packet['focused_evidence'] and '503' in packet['focused_evidence'][0]['excerpt']
    handle = packet['focused_evidence'][0]['handle']
    assert store.get(handle) == source
    assert context.messages == original
    changed, _ = build_review_payload(context, intent.model_copy(update={'outcome': 'Check worker-50'}))
    assert payload.split(',"intention":')[0] == changed.split(',"intention":')[0]
    assert meta['focused_evidence_count'] == 1


@pytest.mark.asyncio
async def test_failed_review_keeps_consumed_calls_without_double_counting(tmp_path, monkeypatch):
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    await model_command(session, '/model router')
    records = []
    async def provider(model, context, options):
        yield EventDone(reason='stop', message=AssistantMessage(
            provider=model.provider, model=model.id, api=model.api, timestamp=1,
            content=[TextContent(text='No decision')], usage=Usage(input=100, output=20, total_tokens=120)))
    with pytest.raises(RuntimeError, match='no valid decision'):
        await review_turn(session._router.selections['max'], AgentContext(system_prompt='', messages=[], tools=[]),
                          stream_fn=provider, get_api_key=session._resolve_api_key,
                          record=lambda name, **kw: records.append(dict(type='custom', customType=name, data=kw['metadata'])))
    stats = model_stats(records)
    assert stats['total'] == 1 and stats['total_tokens'] == 120
    assert stats['models'][0]['purpose'] == 'reviewer'
    assert any(r['customType'] == 'tau.turn_review.failed' for r in records)


def test_price_snapshots_residual_usage_and_legacy_failure_are_not_free():
    base = dict(provider='local', model='model', reasoning='off', level='default',
                pricing=dict(input=2, cache_read=.2, cache_write=0, output=8))
    message = dict(role='assistant', provider='local', model='model',
                   usage=dict(input=100, cache_read=1000, output=20, total_tokens=1120))
    entries = [dict(type='custom', customType='tau.router_selection', data=base),
               dict(type='message', message=message),
               dict(type='custom', customType='tau.turn_review.invocation', data={**base, 'message':message}),
               dict(type='custom', customType='tau.turn_review.completed', data={
                   **base, 'usage_recording':'per_invocation', 'messages':[message]}),
               dict(type='custom', customType='tau.intention.failed', data={'error':'TimeoutError'}),
               dict(type='message', message={**message, 'usage':dict(input=10,output=2,total_tokens=19)})]
    stats = model_stats(entries)
    assert stats['total'] == 3 and stats['total_tokens'] == 2259
    assert stats['estimated_cost'] == pytest.approx(.00112)
    assert stats['priced_calls'] == 2
    assert stats['unrecorded_auxiliary_failures'] == 1
    assert sum(r['unclassified_tokens'] for r in stats['models']) == 7


@pytest.mark.parametrize('pricing,model,complete', [
    ({'input':10,'output':50}, 'different-model', True),
    ({'input':10}, 'm', True),
    ({'input':10,'output':50}, 'm', False),
])
def test_mismatched_partial_or_unknown_usage_cannot_be_priced(pricing, model, complete):
    entries = [dict(type='custom', customType='tau.turn_review.invocation', data={
        'provider':'p', 'model':'m', 'pricing':pricing, 'usage_complete':complete,
        'message':dict(role='assistant', provider='p', model=model,
                       usage=dict(input=100,output=10))})]
    stats = model_stats(entries)
    assert stats['total_tokens'] == 110 and stats['priced_calls'] == 0


@pytest.mark.asyncio
async def test_transport_error_without_usage_is_counted_unpriced(tmp_path, monkeypatch):
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    await model_command(session, '/model router')
    records = []
    async def provider(model, context, options):
        raise RuntimeError('transport failure')
        yield
    with pytest.raises(RuntimeError):
        await review_turn(session._router.selections['max'], AgentContext(system_prompt='',messages=[],tools=[]),
                          stream_fn=provider, get_api_key=session._resolve_api_key,
                          record=lambda name, **kw: records.append(dict(type='custom',customType=name,data=kw['metadata'])))
    stats = model_stats(records)
    assert stats['total'] == 1 and stats['missing_usage'] == 1
    assert stats['priced_calls'] == 0


def test_projection_storage_failure_preserves_full_evidence(monkeypatch):
    from pi_coding_agent.core import review_context
    def unavailable():
        raise OSError('unavailable')
    monkeypatch.setattr(review_context, '_ccr', unavailable)
    text = 'evidence ' * 1000
    context = AgentContext(system_prompt='',messages=[],tools=[])
    context.messages = [dict(role='toolResult',content=[dict(type='text',text=text)])]
    payload, _ = build_review_payload(context)
    assert json.loads(payload)['messages'][0]['message']['content'][0]['text'] == text


def test_already_compressed_evidence_is_not_expanded_for_review(tmp_path, monkeypatch):
    store = CCRStore(str(tmp_path / 'ccr.db'))
    monkeypatch.setattr(active_compression, '_store', store)
    handle = store.put('audit_marker=503\n' + 'heartbeat=200\n' * 1000)
    text = f'[CCR:{handle}] compressed evidence; query audit_marker for details.'
    context = AgentContext(system_prompt='',messages=[],tools=[])
    context.messages = [dict(role='user',content='audit_marker'),
                        dict(role='toolResult',content=[dict(type='text',text=text)])]
    payload, _ = build_review_payload(context)
    packet = json.loads(payload)
    assert packet['messages'][1]['message']['content'][0]['text'] == text
    assert packet['focused_evidence'] == []


def test_review_omits_empty_transport_fields_without_dropping_business_nulls():
    import copy
    import json
    from types import SimpleNamespace
    from pi_coding_agent.core.review_context import build_review_payload
    message = {'role': 'assistant', 'error_message': None, 'content': [
        {'type': 'text', 'text': 'Keep this evidence.', 'text_signature': None},
        {'type': 'toolCall', 'id': 'call1', 'name': 'write',
         'arguments': {'optional': None, 'enabled': False}, 'arguments_raw': None,
         'arguments_repair_applied': False, 'arguments_parse_error': None},
        {'type': 'toolCall', 'id': 'call2', 'name': 'write', 'arguments': {},
         'arguments_raw': '{broken', 'arguments_repair_applied': True,
         'arguments_parse_error': 'invalid JSON'}]}
    before = copy.deepcopy(message)
    payload, _ = build_review_payload(SimpleNamespace(
        system_prompt='', tools=[], messages=[message]))
    actual = json.loads(payload)['messages'][0]['message']
    assert 'error_message' not in actual
    assert 'text_signature' not in actual['content'][0]
    assert actual['content'][0]['text'] == 'Keep this evidence.'
    assert actual['content'][1]['arguments'] == {'optional': None, 'enabled': False}
    assert 'arguments_raw' not in actual['content'][1]
    assert 'arguments_repair_applied' not in actual['content'][1]
    assert actual['content'][2] == before['content'][2]
    assert message == before


def test_auxiliary_stats_use_recorded_level_with_legacy_max_fallback():
    from pi_coding_agent.core.model_stats import model_stats
    base = {'provider': 'p', 'model': 'm', 'reasoning': 'low',
            'message': {'role': 'assistant', 'provider': 'p', 'model': 'm',
                        'usage': {'input': 10, 'output': 2, 'total_tokens': 12}}}
    records = [{'type': 'custom', 'customType': 'tau.intention.invocation',
                'data': {**base, 'level': 'ultra-light'}},
               {'type': 'custom', 'customType': 'tau.intention.invocation', 'data': base}]
    stats = model_stats(records)
    assert stats['total_tokens'] == 24
    assert {row['tier'] for row in stats['models']} == {'ultra-light', 'max'}
