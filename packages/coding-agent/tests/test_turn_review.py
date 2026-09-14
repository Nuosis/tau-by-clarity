"""Functional orchestration checks with controlled provider responses, not quality evals.

Human proof: observe premature ending rejected, the worker edit the real file,
then accept the evidenced ending. Verify Max/effort and read-only review tools.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from pi_agent.types import AgentContext
from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall
from pi_coding_agent.core.turn_review import review_turn
from .test_model_router import make_session, model_command, envelope, metadata, intention_fixture


def done(model, content):
    return EventDone(reason='stop', message=AssistantMessage(
        content=content, api=model.api, provider=model.provider, model=model.id, timestamp=0))


@pytest.mark.asyncio
async def test_rejection_resumes_worker_tools_then_accepts(tmp_path, monkeypatch):
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    target = tmp_path / 'result.txt'
    target.write_text('before')
    worker_calls, review_calls = [], []
    intention_calls = []

    async def provider(model, context, options):
        names = {tool.name for tool in context.tools}
        if 'submit_intention' in names:
            intention_calls.append(model.id)
            yield done(model, [ToolCall(id='intention', name='submit_intention', arguments=intention_fixture())])
        elif 'submit_review' in names:
            review_calls.append(model.id)
            assert model.id == 'max'
            assert options.reasoning == 'high'
            assert names == {'ccr_retrieve', 'submit_review'}
            complete = target.read_text() == 'after'
            args = dict(decision='accept' if complete else 'continue',
                        intention_met=complete, answer_sound=complete,
                        rationale='File content is ' + target.read_text(),
                        evidence_refs=['message:0'],
                        follow_up_requirements=[] if complete else ['Edit result.txt from before to after.'])
            yield done(model, [ToolCall(id='review', name='submit_review', arguments=args)])
        else:
            worker_calls.append(model.id)
            actions = []
            if len(worker_calls) == 2:
                assert 'Completion review requires further work' in str(context.messages)
                actions = [{'name': 'edit', 'arguments': {'path': str(target),
                           'oldText': 'before', 'newText': 'after'}}]
            yield done(model, [ToolCall(id='worker' + str(len(worker_calls)), name='submit_response',
                       arguments=envelope(actions, metadata() if actions else None))])

    session._provider_stream = provider
    await model_command(session, '/model router')
    await session.prompt('Change result.txt from before to after.')
    assert session.agent.state.error is None
    assert target.read_text() == 'after'
    assert len(worker_calls) == 3
    assert review_calls == ['max', 'max']
    assert intention_calls == ['ultra-light']
    await session.prompt('Report the current result without modifying it.')
    assert intention_calls == ['ultra-light', 'max']
    from pathlib import Path
    entries = [json.loads(line) for line in Path(session._session_manager.get_session_file()).read_text().splitlines()]
    assert [e['data']['level'] for e in entries
            if e.get('customType') == 'tau.intention.invocation'] == ['ultra-light', 'max']
    assert review_calls == ['max', 'max', 'max']
    assert target.read_text() == 'after'
    assert session.intention_placeholder == intention_fixture()['outcome']
    assert 'edit' in {tool.name for tool in session.agent.state.tools}


@pytest.mark.asyncio
@pytest.mark.parametrize('queue', ['steer', 'follow_up'])
async def test_user_correction_revises_intention_without_injecting_it(tmp_path, monkeypatch, queue):
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    intentions, reviews, workers = [], [], []
    async def provider(model, context, options):
        names = {t.name for t in context.tools}
        if 'submit_intention' in names:
            value = {**intention_fixture(), 'outcome': f'INTENT_ONLY_{len(intentions)}'}
            intentions.append(value)
            yield done(model, [ToolCall(id='intent', name='submit_intention', arguments=value)])
        elif 'submit_review' in names:
            content = context.messages[0].content
            packet = json.loads(content if isinstance(content, str) else content[0].text)
            reviews.append(packet['intention'])
            yield done(model, [ToolCall(id='review', name='submit_review', arguments={
                'decision': 'accept', 'rationale': 'Answered the current request.',
                'intention_met': True, 'answer_sound': True,
                'evidence_refs': ['message:0'], 'follow_up_requirements': []})])
        else:
            assert 'INTENT_ONLY_' not in str(context.messages)
            workers.append(context)
            if len(workers) == 1:
                await getattr(session, queue)('Correction: report only; do not edit anything.')
            yield done(model, [ToolCall(id='worker', name='submit_response', arguments=envelope([], None))])
    session._provider_stream = provider
    await model_command(session, '/model router')
    await session.prompt('Inspect the fixture.')
    assert session.agent.state.error is None
    assert len(intentions) == 2 and len(workers) == 2
    assert reviews == [intentions[-1]]
    assert session.intention_placeholder == 'INTENT_ONLY_1'
    await session.switch_session(session._session_manager.get_session_file())
    assert session.intention_placeholder == 'INTENT_ONLY_1'
    assert all('INTENT_ONLY_' not in str(m) for m in session.agent.state.messages)


@pytest.mark.parametrize('met,sound', [(False, True), (True, False), (False, False)])
def test_cannot_accept_unmet_intention_or_unsound_answer(met, sound):
    from pydantic import ValidationError
    from pi_coding_agent.core.turn_review import IntentionReviewDecision
    with pytest.raises(ValidationError):
        IntentionReviewDecision(decision='accept', rationale='Done', evidence_refs=['message:0'],
            follow_up_requirements=[], intention_met=met, answer_sound=sound)


@pytest.mark.asyncio
async def test_reviewer_can_retrieve_real_ccr_evidence(tmp_path, monkeypatch):
    from pi_coding_agent import active_compression
    from pi_coding_agent.active_compression.ccr import CCRStore
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    await model_command(session, '/model router')
    store = CCRStore(str(tmp_path / 'ccr.db'))
    monkeypatch.setattr(active_compression, '_store', store)
    handle = store.put('Deployment verification: release 123 failed with HTTP 503.')
    calls, events = [], []

    async def provider(model, context, options):
        calls.append(context)
        if len(calls) == 1:
            yield done(model, [ToolCall(id='lookup', name='ccr_retrieve',
                       arguments={'handle': handle, 'query': 'Deployment verification release 123'})])
        else:
            assert '503' in str(context.messages)
            yield done(model, [ToolCall(id='verdict', name='submit_review', arguments={
                'decision': 'continue', 'rationale': 'Deployment failed with HTTP 503.',
                'evidence_refs': [handle], 'follow_up_requirements': ['Investigate the failed deployment.']})])

    result = await review_turn(session._router.selections['max'],
        AgentContext(system_prompt='Verify deployment.', messages=[{'role': 'user', 'timestamp': 0,
            'content': f'Deployment output compressed [CCR:{handle}]. Candidate: deployed successfully.'}], tools=[]),
        stream_fn=provider, get_api_key=session._resolve_api_key,
        record=lambda name, **kw: events.append((name, kw)))
    assert result.decision == 'continue'
    assert len(calls) == 2
    assert events[-1][1]['metadata']['retrievals'][0]['arguments']['handle'] == handle


@pytest.mark.asyncio
async def test_no_verdict_is_not_silent_approval(tmp_path, monkeypatch):
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    await model_command(session, '/model router')
    async def provider(model, context, options):
        yield done(model, [TextContent(text='Looks fine.')])
    with pytest.raises(RuntimeError, match='no valid decision'):
        await review_turn(session._router.selections['max'],
            AgentContext(system_prompt='', messages=[], tools=[]), stream_fn=provider,
            get_api_key=session._resolve_api_key, record=lambda *a, **k: None)


@pytest.mark.asyncio
@pytest.mark.parametrize('establish', [False, True])
async def test_cancel_stops_review_without_approval(tmp_path, monkeypatch, establish):
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    await model_command(session, '/model router')
    started, cancelled, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def provider(model, context, options):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
        yield done(model, [TextContent(text='Unreachable')])
    task = asyncio.create_task(review_turn(session._router.selections['max'],
        AgentContext(system_prompt='', messages=[], tools=[]), stream_fn=provider,
        get_api_key=session._resolve_api_key, record=lambda *a, **k: None,
        cancel_event=cancelled, establish_intention=establish))
    await started.wait()
    cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
async def test_session_review_uses_persisted_compression_and_custom_memory(tmp_path, monkeypatch, typed):
    from .test_persisted_ccr_context import _large_log, _large_tool_message
    from pi_coding_agent.active_compression.persisted_context import compress_message_for_persistence
    from pi_coding_agent.active_compression.ccr import CCRStore
    from pi_coding_agent import active_compression
    from pi_coding_agent.core.messages import CustomMessage
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    monkeypatch.setattr(active_compression, '_store', CCRStore(str(tmp_path / 'ccr.db')))
    raw = _large_tool_message(_large_log())
    raw["details"] = {"truncation": {"content": _large_log()}, "exitCode": 1}
    raw["is_error"] = True
    compressed, metadata = compress_message_for_persistence(raw)
    assert metadata
    session._session_manager.append_message(compressed, active_compression=metadata)
    memory = CustomMessage(custom_type='memory_recall', content='Prefer verification over assumptions.', display=False)
    from pi_ai.types import ToolResultMessage
    await session._transform_context([ToolResultMessage.model_validate(raw) if typed else raw, memory])
    await model_command(session, '/model router')
    import copy
    worker_before = copy.deepcopy(session._review_context_messages)
    calls = []
    async def provider(model, context, options):
        calls.append(context)
        content = context.messages[0].content
        packet = json.loads(content if isinstance(content, str) else content[0].text)
        rendered = json.dumps(packet)
        assert json.dumps(_large_log())[1:-1] not in rendered
        assert metadata['refs'][0]['handle'] in rendered
        evidence = packet['messages'][0]['message']
        assert 'details' not in evidence
        assert evidence['tool_call_id'] == raw['tool_call_id']
        assert evidence['is_error'] is True
        assert [(b['type'], b.get('text')) for b in evidence['content']] == [
            (b['type'], b.get('text')) for b in compressed['content']
        ]
        assert raw['details']['truncation']['content'] == _large_log()
        assert 'Prefer verification over assumptions.' in rendered
        assert 'edit' in {tool['name'] for tool in packet['worker_capabilities']}
        assert packet['candidate'] == 'message:2'
        if len(calls) == 1:
            yield done(model, [ToolCall(id='retrieve', name='ccr_retrieve', arguments={
                'handle': metadata['refs'][0]['handle'], 'query': 'auth_middleware_unique_token'})])
            return
        retrieved = context.messages[2]
        assert retrieved.role == 'toolResult' and not retrieved.is_error
        assert retrieved.details['kept_items'] > 0
        assert 'worker-42 persisted context auth_middleware_unique_token processing queue' in retrieved.content[0].text
        yield done(model, [ToolCall(id='review', name='submit_review', arguments={
            'decision': 'accept', 'rationale': 'Verified.', 'evidence_refs': ['message:2'],
            'follow_up_requirements': []})])
    session._provider_stream = provider
    candidate = done(session.agent.state.model, [TextContent(text='Complete.')]).message
    context = AgentContext(system_prompt='Task instructions', messages=[], tools=session.agent.state.tools)
    await session._prepare_next_turn({'context': context, 'message': candidate, 'tool_results': []})

    assert len(calls) == 2
    assert session._review_context_messages == worker_before
