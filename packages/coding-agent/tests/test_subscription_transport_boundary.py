"""Verify subscription selection through outgoing HTTP, without provider traffic."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pi_ai.types import Context, UserMessage, SimpleStreamOptions
from pi_ai.stream import stream_simple
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_registry import ModelRegistry


class SubscriptionBoundaryTests(unittest.TestCase):
    def storage(self, *, expired=False):
        return AuthStorage.in_memory({
            'openai': {'type': 'oauth', 'access_token': 'subscription-placeholder',
                       'expires_at': 1 if expired else time.time() + 3600},
            'api_keys': {'openai': 'paid-key-placeholder'},
        })

    def registry(self, auth, config=None):
        path = Path(self.tmp.name) / 'models.json'
        path.write_text(json.dumps(config or {"providers": {}}))
        return ModelRegistry(auth, str(path))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_configured_and_new_models_select_subscription(self):
        r = self.registry(self.storage())
        for name in ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-6-astra', 'gpt-future-model']:
            with self.subTest(model=name):
                model = r.find('openai', name)
                self.assertEqual(model.id, name)
                self.assertEqual(model.api, 'openai-codex-responses')
                self.assertEqual(model.base_url, 'https://chatgpt.com/backend-api')

    def test_subscription_wins_over_models_json_api_key(self):
        r = self.registry(self.storage(), {'providers': {'openai': {'apiKey': 'custom-paid-key', 'baseUrl': 'https://api.openai.com/v1'}}})
        self.assertEqual(r.get_api_key('openai'), 'subscription-placeholder')

    def test_expired_subscription_never_falls_back(self):
        auth = self.storage(expired=True)
        with patch.object(auth, '_refresh_oauth_token', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'subscription'):
                auth.resolve_api_key('openai')
            with self.assertRaisesRegex(RuntimeError, 'subscription'):
                asyncio.run(auth.resolve_api_key_async('openai'))

    def test_api_key_only_configuration_remains_api(self):
        auth = AuthStorage.in_memory({'openai': {'type': 'api_key', 'key': 'paid-key-placeholder'}})
        r = self.registry(auth)
        self.assertEqual(r.find('openai', 'gpt-6-astra').api, 'openai-responses')
        self.assertEqual(r.get_api_key('openai'), 'paid-key-placeholder')

    def test_provider_rejection_stays_on_subscription_endpoint(self):
        async def scenario():
            seen = []
            def handler(request):
                seen.append((str(request.url), request.headers['authorization'], json.loads(request.content)))
                return httpx.Response(403, text='Model unavailable for this subscription')
            client_type = httpx.AsyncClient
            def client(**kwargs):
                return client_type(transport=httpx.MockTransport(handler), **kwargs)
            r = self.registry(self.storage())
            model = r.find('openai', 'gpt-6-astra')
            with patch('httpx.AsyncClient', side_effect=client):
                stream = stream_simple(model, Context(messages=[UserMessage(content='offline test', timestamp=0)]),
                                                       SimpleStreamOptions(api_key=r.get_api_key('openai')))
                events = [event async for event in stream]
            self.assertEqual(len(seen), 1)
            url, authorization, body = seen[0]
            self.assertEqual(url, 'https://chatgpt.com/backend-api/codex/responses')
            self.assertEqual(authorization, 'Bearer subscription-placeholder')
            self.assertEqual(body['model'], 'gpt-6-astra')
            self.assertTrue(events)
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
