"""Opt-in live Max reviewer with scripted worker and actual sandbox file tools.

Not a broad autonomous-worker quality eval. The finite worker replay exposes an
unevidenced ending, edits/verifies the requested file, then proposes completion.
"""
import json
import os
from pathlib import Path

import pytest
from pi_ai.stream import stream_simple
from pi_ai.types import ToolCall
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_registry import ModelRegistry
from pi_coding_agent.core.router_config import load_router_selections
from pi_coding_agent.config import get_models_path
from .test_model_router import make_session, model_command, envelope, metadata
from .test_turn_review import done


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get('TAU_LIVE_INTENTION'), reason='Opt-in provider spend')
async def test_live_max_intention_reject_work_verify_accept(tmp_path, monkeypatch):
    registry = ModelRegistry(AuthStorage())
    maximum = load_router_selections(get_models_path(), registry)['max']
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    target = tmp_path / 'result.txt'
    target.write_text('before\npreserve\n')
    original_resolve = session._resolve_api_key
    async def resolve(provider):
        if provider == maximum.model.provider:
            return registry.get_api_key(provider)
        return await original_resolve(provider)
    session._resolve_api_key = resolve
    worker_calls = []
    async def provider(model, context, options):
        if any(t.name in {'submit_intention', 'submit_review'} for t in context.tools):
            assert model.id == maximum.model.id
            async for event in stream_simple(model, context, options):
                yield event
            return
        worker_calls.append(len(worker_calls) + 1)
        step = len(worker_calls)
        if step == 1:
            wire = envelope([], None)
        elif step == 2:
            wire = envelope([{'name': 'edit', 'arguments': {'path': str(target),
                             'oldText': 'before', 'newText': 'after'}}], metadata())
        elif step == 3:
            wire = envelope([{'name': 'read', 'arguments': {'path': str(target),
                             'offset': None, 'limit': None}}], metadata())
        elif step == 4:
            wire = envelope([], None)
            wire['text'] = 'The saved file still contains before. No replacement was made.'
        elif step == 5:
            wire = envelope([], None)
            wire['text'] = 'Changed result.txt from before to after; read it back and confirmed the preserve line is unchanged.'
        else:
            raise AssertionError('Live reviewer rejected the completed finite worker replay')
        yield done(model, [ToolCall(id=f'worker{step}', name='submit_response', arguments=wire)])
    session._provider_stream = provider
    await model_command(session, '/model router')
    session._router.selections['max'] = maximum
    await session.prompt(f'In {target}, change before to after, preserve the other line, and verify the saved file.')
    entries = [e.data for e in session._session_manager.get_entries()]
    artifact = Path(os.environ['TAU_LIVE_INTENTION'])
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / 'session.json').write_text(json.dumps(entries, indent=2, default=str))
    assert session.agent.state.error is None
    assert target.read_text() == 'after\npreserve\n'
    intents = [e for e in entries if e.get('customType') == 'tau.intention.completed']
    reviews = [e['data']['decision'] for e in entries if e.get('customType') == 'tau.turn_review.completed']
    assert len(intents) == 1
    assert [r['decision'] for r in reviews] == ['continue', 'continue', 'accept']
    assert reviews[1]['intention_met'] and not reviews[1]['answer_sound']
    assert reviews[-1]['intention_met'] and reviews[-1]['answer_sound']
    print(json.dumps({'model': maximum.model.id, 'reasoning': maximum.reasoning,
                      'intention': session.intention_placeholder, 'reviews': reviews,
                      'worker_calls': len(worker_calls), 'artifact': str(artifact)}))
