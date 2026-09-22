import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from pi_coding_agent.core import cli_debug_log
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.modes.rpc.login import RpcLoginController


@pytest.mark.asyncio
@pytest.mark.parametrize('error_code', ['invalid_grant', 'SECRET-IN-ERROR-FIELD'])
async def test_native_exchange_failure_is_diagnosable_without_credentials(tmp_path, monkeypatch, error_code):
    path = tmp_path / 'diagnostic.jsonl'
    monkeypatch.setattr(cli_debug_log, '_LOG_PATH', str(path))
    import pi_ai.utils.oauth.anthropic as provider
    client_type = httpx.AsyncClient
    observed = []
    def respond(request):
        observed.append(json.loads(request.content))
        return httpx.Response(400, json={
            'error': error_code, 'error_description': 'SECRET-IN-BODY',
            'access_token': 'SECRET-TOKEN',
        })
    monkeypatch.setattr(provider.httpx, 'AsyncClient', lambda: client_type(transport=httpx.MockTransport(respond)))
    auth = AuthStorage.in_memory()
    events = []
    controller = RpcLoginController(events.append)
    controller.start(SimpleNamespace(session_id='diagnostic-session', auth_storage=auth), provider='anthropic', method='subscription')
    while not any(e['event'] == 'request' for e in events):
        await asyncio.sleep(0)
    request = next(e for e in events if e['event'] == 'request')
    controller.respond(login_id=request['loginId'], request_id=request['requestId'], value='SECRET-CODE#SECRET-STATE')
    await controller._task
    assert observed[0]['code'] == 'SECRET-CODE'
    assert events[-1]['event'] == 'failed'
    assert auth.get_oauth_token('anthropic') is None
    records = [json.loads(line) for line in path.read_text().splitlines()]
    failure = next(r for r in records if r['event'] == 'native_login_failed')
    assert failure['stage'] == 'subscription'
    assert failure['failure_code'] == ('invalid_grant' if error_code == 'invalid_grant' else 'unknown')
    assert any(frame['function'] == 'login_anthropic' for frame in failure['frames'])
    assert any(r['event'] == 'native_login_response' for r in records)
    assert 'SECRET' not in path.read_text()
    assert 'SECRET' not in json.dumps(events)
    assert 'https://' not in path.read_text()

@pytest.mark.asyncio
async def test_native_success_still_persists_and_diagnostics_remain_private(tmp_path, monkeypatch):
    path = tmp_path / 'diagnostic.jsonl'
    monkeypatch.setattr(cli_debug_log, '_LOG_PATH', str(path))
    import pi_ai.utils.oauth.anthropic as provider
    client_type = httpx.AsyncClient
    monkeypatch.setattr(provider.httpx, 'AsyncClient', lambda: client_type(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={
            'access_token': 'SECRET-ACCESS', 'refresh_token': 'SECRET-REFRESH', 'expires_in': 3600,
        })
    )))
    auth_path = str(tmp_path / 'auth.json')
    events = []
    controller = RpcLoginController(events.append)
    controller.start(SimpleNamespace(session_id='diagnostic-session', auth_storage=AuthStorage.create(auth_path)),
                     provider='anthropic', method='subscription')
    while not any(e['event'] == 'request' for e in events):
        await asyncio.sleep(0)
    request = next(e for e in events if e['event'] == 'request')
    controller.respond(login_id=request['loginId'], request_id=request['requestId'], value='SECRET-CODE#SECRET-STATE')
    await controller._task
    assert events[-1]['event'] == 'completed'
    assert AuthStorage.create(auth_path).get_oauth_token('anthropic')['access_token'] == 'SECRET-ACCESS'
    assert 'SECRET' not in path.read_text()
    assert 'https://' not in path.read_text()
