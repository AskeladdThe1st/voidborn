import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.models import REGISTRY, Registry
from backend.runtime import OllamaRuntime, RuntimeErrorLocal
from test_app import FakeRuntime


class SelectionTests(unittest.TestCase):
    def test_discovers_and_runs_unregistered_local_model(self):
        tag = 'local-test:latest'
        tags = []

        def handler(request):
            path = request.url.path
            if path == '/api/tags':
                return httpx.Response(200, json={'models': tags})
            if path == '/api/show':
                self.assertEqual(json.loads(request.content), {'model': tag})
                return httpx.Response(200, json={'model_info': {'general.architecture': 'test'}})
            if path == '/api/ps':
                return httpx.Response(200, json={'models': []})
            self.assertEqual(path, '/api/chat')
            self.assertEqual(json.loads(request.content)['model'], tag)
            return httpx.Response(200, text='{"message":{"content":"Hi"}}\n{"done":true}\n')

        runtime = OllamaRuntime()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / 'chat.sqlite3', runtime)) as client:
                self.assertFalse(any(m['installed'] for m in client.get('/api/models').json()['models']))
                tags.extend([{'name': tag}, {'name': 'remote:latest', 'remote_host': 'example.com'}])
                installed = [m for m in client.get('/api/models').json()['models'] if m['installed']]
                self.assertEqual([m['displayName'] for m in installed], [tag])
                model_id = installed[0]['id']
                self.assertTrue(client.get('/api/status', params={'model_id': model_id}).json()['ready'])
                cid = client.post('/api/conversations').json()['id']
                message = {'content': 'Hello', 'model_id': model_id}
                response = client.post(f'/api/conversations/{cid}/messages', json=message)
                self.assertEqual(json.loads(response.text.splitlines()[-1])['type'], 'done')
                comparison = client.post('/api/compare', json=message)
                self.assertEqual(json.loads(comparison.text.splitlines()[0])['model_id'], model_id)
                self.assertEqual(json.loads(comparison.text.splitlines()[-2])['type'], 'result')
                tags.clear()
                self.assertEqual(client.get('/api/status', params={'model_id': model_id}).status_code, 422)

    def test_comparison_skips_missing_models_and_reports_no_models(self):
        class InstalledRuntime(FakeRuntime):
            none_installed = False

            async def catalog(self):
                catalog = await super().catalog()
                for entry in catalog['models']:
                    entry['installed'] = not self.none_installed and entry['id'] == 'gemma3-4b'
                return catalog

        runtime = InstalledRuntime()
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / 'chat.sqlite3', runtime)) as client:
                events = [json.loads(line) for line in client.post('/api/compare', json={'content': 'Hello'}).text.splitlines()]
                self.assertEqual([e['model_id'] for e in events if e['type'] == 'start'], ['gemma3-4b'])
                runtime.none_installed = True
                self.assertEqual(client.post('/api/compare', json={'content': 'Hello'}).status_code, 409)
                runtime.none_installed = False
                self.assertEqual(client.post('/api/compare', json={'content': 'Hello'}).status_code, 200)

    def test_switch_preserves_history_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chat.sqlite3'
            runtime = FakeRuntime()
            with TestClient(create_app(path, runtime)) as client:
                cid = client.post('/api/conversations').json()['id']
                url = '/api/conversations/' + cid
                for selected in [m.id for m in REGISTRY.models]:
                    response = client.post(url + '/messages', json={'content': 'Hello', 'model_id': selected})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(runtime.selected, selected)
                self.assertEqual(len(client.get(url).json()['messages']), 2 * len(REGISTRY.models))
                self.assertEqual(client.post(url + '/messages', json={'content': 'Hi', 'model_id': 'unknown'}).status_code, 422)
                self.assertEqual(len(client.get(url).json()['messages']), 2 * len(REGISTRY.models))
                self.assertEqual(len(client.get('/api/models').json()['models']), len(REGISTRY.models))
            with TestClient(create_app(path, runtime)) as client:
                self.assertEqual(len(client.get(url).json()['messages']), 2 * len(REGISTRY.models))

    def test_comparison_fresh_context_failure_isolation_no_database_writes(self):
        calls = []

        class ComparedRuntime(FakeRuntime):
            async def stream(self, messages, model_id=None):
                calls.append((model_id, messages))
                if model_id == 'qwen3-8b':
                    raise RuntimeErrorLocal('Test failure')
                yield 'A test answer'

        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / 'chat.sqlite3', ComparedRuntime())) as client:
                response = client.post('/api/compare', json={'content': 'Same question'})
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual([e['type'] for e in events], ['start', 'model_error'] + ['start', 'delta', 'result'] * (len(REGISTRY.models) - 1) + ['done'])
                self.assertEqual([c[0] for c in calls], [m.id for m in REGISTRY.models])
                self.assertEqual(calls[0][1], calls[1][1])
                self.assertEqual(calls[0][1], [{'role': 'user', 'content': 'Same question'}])
                self.assertGreaterEqual(events[-2]['seconds'], 0)
                self.assertEqual(client.get('/api/conversations').json(), [])

    def test_registry_rejects_remote_tags_and_duplicate_ids(self):
        config = REGISTRY.model_dump()
        config['models'][0]['ollamaModel'] = 'https://external.example/model'
        with self.assertRaises(ValueError):
            Registry.model_validate(config)
        config = REGISTRY.model_dump()
        config['models'][1]['id'] = config['models'][0]['id']
        with self.assertRaises(ValueError):
            Registry.model_validate(config)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_name_detection_refreshes_after_install(self):
        selected = REGISTRY.get('qwen35-9b')
        self.assertEqual(selected.ollamaModel, 'hf.co/tutuchen2000/Qwen3.5-9B-abliterated-GGUF:Q4_K_M')
        self.assertEqual(selected.displayName, 'Qwen3.5 9B Abliterated - Q4_K_M')
        installed = [{'name': selected.displayName, 'model': selected.ollamaModel}]
        def handler(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:11434/api/tags')
            return httpx.Response(200, json={'models': installed})
        runtime = OllamaRuntime()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        self.assertFalse((await runtime.status(selected.id))['ready'])
        installed[:] = [{'name': selected.ollamaModel, 'model': 'ignored'}]
        self.assertTrue((await runtime.status(selected.id))['ready'])
        installed.clear()
        self.assertFalse((await runtime.status(selected.id))['ready'])

    async def test_qwen35_uses_native_thinking_and_sampling_by_default(self):
        selected = REGISTRY.get('qwen35-9b')
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['model'], selected.ollamaModel)
            self.assertNotIn('think', body)
            self.assertEqual(body['messages'], [{'role': 'user', 'content': '  Original message.\n'}])
            self.assertNotIn('num_gpu', body['options'])
            self.assertNotIn('temperature', body['options'])
            self.assertNotIn('top_p', body['options'])
            return httpx.Response(200, text='{"message":{"content":"Hello"}}\n{"done":true}\n')
        runtime = OllamaRuntime()
        runtime.prepare = AsyncMock()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        self.assertEqual([part async for part in runtime.stream([{'role': 'user', 'content': '  Original message.\n'}], selected.id)], ['Hello'])

    async def test_explicit_system_prompt_is_passed_verbatim(self):
        configured = REGISTRY.model_copy(update={'system_instruction': 'Custom system prompt.\n'})
        messages = [{'role': 'user', 'content': 'My message.'}]

        def handler(request):
            self.assertEqual(json.loads(request.content)['messages'], [
                {'role': 'system', 'content': configured.system_instruction}, *messages,
            ])
            return httpx.Response(200, text='{"message":{"content":"Hello"},"done":true}\n')

        runtime = OllamaRuntime()
        runtime.prepare = AsyncMock()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        with patch('backend.runtime.REGISTRY', configured):
            self.assertEqual([part async for part in runtime.stream(messages)], ['Hello'])

    async def test_unloads_other_model_before_generation_and_uses_model_options(self):
        selected = REGISTRY.get('gemma3-4b')
        other = REGISTRY.get('qwen3-8b')
        loaded, calls = [other.ollamaModel], []

        def handler(request):
            self.assertEqual(request.url.host, '127.0.0.1')
            path = request.url.path
            calls.append(path)
            if path == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': selected.ollamaModel}]})
            if path == '/api/show':
                return httpx.Response(200, json={'model_info': {'general.architecture': 'gemma3'}})
            if path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': name} for name in loaded]})
            body = json.loads(request.content)
            if path == '/api/generate':
                self.assertEqual(body, {'model': other.ollamaModel, 'keep_alive': 0, 'stream': False})
                loaded.clear()
                return httpx.Response(200, json={'done': True})
            self.assertEqual(path, '/api/chat')
            self.assertFalse(loaded)
            self.assertEqual(body['model'], selected.ollamaModel)
            self.assertEqual(body['keep_alive'], 0)
            self.assertEqual(body['options'], selected.options.model_dump(exclude_none=True))
            self.assertNotIn('think', body)
            self.assertEqual(body['messages'], [])
            return httpx.Response(200, text='{"message":{"content":"Hi"}}\n{"done":true}\n')

        runtime = OllamaRuntime()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        self.assertEqual([part async for part in runtime.stream([], selected.id)], ['Hi'])
        self.assertLess(calls.index('/api/generate'), calls.index('/api/chat'))

    async def test_remote_model_metadata_rejected_before_prompt(self):
        selected = REGISTRY.get()
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': selected.ollamaModel}]})
            return httpx.Response(200, json={'remote_host': 'https://example.com', 'model_info': {'x': 1}})

        runtime = OllamaRuntime()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        with self.assertRaises(RuntimeErrorLocal):
            _ = [part async for part in runtime.stream([])]
        self.assertNotIn('/api/chat', calls)
