"""Exercise the Jev Decisions request and typed response over local HTTP."""
import pytest
from aiohttp import web
from pi_coding_agent.core.intention_alignment import judge_intention_alignment
from pi_coding_agent.core.turn_review import Intention


@pytest.mark.asyncio
@pytest.mark.parametrize('choice', ['continues_active_intention', 'starts_new_intention'])
async def test_jev_judges_latest_input_against_active_intention(choice):
    requests = []
    async def decisions(request):
        assert request.headers['Authorization'] == 'Bearer test-key'
        payload = await request.json()
        requests.append(payload)
        return web.json_response({'model': 'typesafe/jev-test', 'answers': {
            'alignment': {'type': 'choice', 'choice': choice,
                          'probabilities': {'continues_active_intention': 0.9,
                                            'starts_new_intention': 0.1}, 'confidence': 0.9}}})

    app = web.Application()
    app.router.add_post('/api/alpha/decisions', decisions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    endpoint = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/api/alpha/decisions'
    async def key(provider):
        assert provider == 'openrouter'
        return 'test-key'
    events = []
    intention = Intention(outcome='Explain and repair the intention flow.',
                          completion_evidence=['The flow is verified.'], scope='Tau intention flow')
    try:
        result = await judge_intention_alignment(
            intention, 'Use Jev for the latest request.', get_api_key=key,
            record=lambda name, **kw: events.append((name, kw['metadata'])), endpoint=endpoint)
    finally:
        await runner.cleanup()
    assert result == choice
    assert len(requests) == 1
    assert requests[0]['model'] == '~typesafe/jev-latest'
    assert requests[0]['state'] == {'active_intention': intention.model_dump(),
                                    'latest_input': 'Use Jev for the latest request.'}
    assert requests[0]['questions']['alignment']['type'] == 'choice'
    assert events[-1][0] == 'tau.intention_alignment.completed'
    assert events[-1][1]['choice'] == choice
    assert 'test-key' not in str(events)
