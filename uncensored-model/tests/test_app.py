import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from backend.models import MAX_PROMPT_BYTES, REGISTRY

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.runtime import OllamaRuntime, RuntimeErrorLocal, context_messages
from backend.storage import Store


class FakeRuntime:
    fail = False
    contexts = None

    async def catalog(self):
        return {'default_model': REGISTRY.default_model, 'available': True, 'models': [{**m.model_dump(), 'installed': True} for m in REGISTRY.models]}

    async def status(self, model_id=None):
        return {'ready': True}

    async def stream(self, messages, model_id=None):
        self.contexts = messages
        self.selected = model_id
        yield 'Hello '
        if self.fail:
            raise RuntimeErrorLocal('Runtime stopped')
        yield '世界'


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'chat.sqlite3'
        self.runtime = FakeRuntime()
        self.client = TestClient(create_app(self.path, self.runtime))
        self.client.__enter__()
        self.cid = self.client.post('/api/conversations').json()['id']
        self.url = '/api/conversations/' + self.cid

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_stream_history_restart_rename_delete(self):
        response = self.client.post(self.url + '/messages', json={'content': 'Hi'})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e['type'] for e in events], ['start', 'delta', 'delta', 'done'])
        self.assertEqual(self.runtime.contexts, [{'role': 'user', 'content': 'Hi'}])
        self.client.patch(self.url, json={'title': 'Saved chat'})
        with TestClient(create_app(self.path, self.runtime)) as restarted:
            chat = restarted.get(self.url).json()
            self.assertEqual(chat['title'], 'Saved chat')
            self.assertEqual(chat['messages'][1]['content'], 'Hello 世界')
            self.assertEqual(chat['messages'][1]['status'], 'complete')
            self.assertEqual(restarted.delete(self.url).status_code, 204)
            self.assertEqual(restarted.get(self.url).status_code, 404)
        self.assertEqual(Store(self.path).messages(self.cid), [])

    def test_failure_preserves_partial_and_retry_does_not_duplicate_user(self):
        self.runtime.fail = True
        response = self.client.post(self.url + '/messages', json={'content': 'Hi'})
        self.assertIn('Runtime stopped', response.text)
        messages = self.client.get(self.url).json()['messages']
        self.assertEqual(messages[-1]['content'], 'Hello ')
        self.assertEqual(messages[-1]['status'], 'error')
        self.assertEqual(self.client.post(self.url + '/messages', json={'content': 'Again'}).status_code, 409)
        self.runtime.fail = False
        self.client.post(self.url + '/retry')
        messages = self.client.get(self.url).json()['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]['status'], 'complete')

    def test_restart_marks_unfinished_stream(self):
        self.client.post(self.url + '/messages', json={'content': 'Hi'})
        store = Store(self.path)
        mid = store.messages(self.cid)[-1]['id']
        store.update_message(mid, 'Partial', 'streaming')
        store.initialize()
        self.assertEqual(store.messages(self.cid)[-1]['status'], 'interrupted')

    def test_validation_and_cross_site_protection(self):
        self.assertEqual(self.client.post(self.url + '/messages', json={'content': ' '}).status_code, 422)
        for content in ['a' * (MAX_PROMPT_BYTES + 1), '世' * (MAX_PROMPT_BYTES // 3 + 1)]:
            self.assertEqual(self.client.post(self.url + '/messages', json={'content': content}).status_code, 413)
        self.assertEqual(self.client.patch(self.url, json={'title': ' '}).status_code, 422)
        self.assertEqual(self.client.post('/api/conversations', headers={'Origin': 'https://example.com'}).status_code, 403)
        self.assertEqual(self.client.get('/api/conversations', headers={'Host': 'evil.example'}).status_code, 400)
        self.assertEqual(self.client.get('/api/conversations', headers={'Sec-Fetch-Site': 'cross-site'}).status_code, 403)

    def test_context_keeps_recent_complete_turns(self):
        messages = [{'role': 'user', 'content': 'old' * 1000, 'status': 'complete'},
                    {'role': 'assistant', 'content': 'answer', 'status': 'complete'},
                    {'role': 'user', 'content': 'new', 'status': 'complete'},
                    {'role': 'assistant', 'content': '', 'status': 'streaming'}]
        omitted = []
        self.assertEqual(context_messages(messages, 2800, omitted=omitted), [{'role': 'user', 'content': 'new'}])
        self.assertEqual(omitted, [1])

    def test_long_multiline_prompt_is_stored_and_sent_intact(self):
        code = '    first_line = True\n' + '    # multiline code\n' * 900 + '\n'
        response = self.client.post(self.url + '/messages', json={'content': code})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.runtime.contexts[-1]['content'], code)
        self.assertEqual(self.client.get(self.url).json()['messages'][0]['content'], code)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_one_generation_and_no_delete_during_stream(self):
        started, release = asyncio.Event(), asyncio.Event()

        class WaitingRuntime(FakeRuntime):
            async def stream(self, messages, model_id=None):
                started.set()
                await release.wait()
                yield 'Done'

        with tempfile.TemporaryDirectory() as folder:
            app = create_app(Path(folder) / 'chat.sqlite3', WaitingRuntime())
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
                    cid = (await client.post('/api/conversations')).json()['id']
                    url = '/api/conversations/' + cid
                    first = asyncio.create_task(client.post(url + '/messages', json={'content': 'Hi'}))
                    try:
                        await asyncio.wait_for(started.wait(), timeout=10)
                        self.assertEqual((await client.post(url + '/messages', json={'content': 'Again'})).status_code, 409)
                        self.assertEqual((await client.post('/api/compare', json={'content': 'Compare'})).status_code, 409)
                        self.assertEqual((await client.delete(url)).status_code, 409)
                    finally:
                        release.set()
                        await first
                    self.assertEqual((await client.delete(url)).status_code, 204)

    async def test_ndjson_adapter_and_local_destination(self):
        def handler(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:11434/api/chat')
            payload = json.loads(request.content)
            self.assertNotIn('think', payload)
            self.assertTrue(payload['stream'])
            return httpx.Response(200, text='{"message":{"content":"Hi"},"done":false}\n{"done":true}\n')
        runtime = OllamaRuntime()
        runtime.prepare = AsyncMock()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        self.assertEqual([part async for part in runtime.stream([])], ['Hi'])

    async def test_truncated_stream_is_failure(self):
        runtime = OllamaRuntime()
        runtime.prepare = AsyncMock()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(lambda _: httpx.Response(200, text='{"message":{"content":"partial"}}\n')))
        with self.assertRaises(RuntimeErrorLocal):
            _ = [part async for part in runtime.stream([])]

    async def test_gguf_split_think_prefix(self):
        def handler(request):
            self.assertEqual(json.loads(request.content)['messages'], [])
            events = [{'message': {'content': part}} for part in ['</th', 'ink>', 'Hello']]
            return httpx.Response(200, text='\n'.join(json.dumps(e) for e in [*events, {'done': True}]))
        runtime = OllamaRuntime()
        runtime.prepare = AsyncMock()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        self.assertEqual(''.join([part async for part in runtime.stream([])]), 'Hello')

    async def test_reasoning_tags_inside_code_are_preserved(self):
        chunks = ['<thi', 'nk>hidden</think>Code: `<thi', 'nk>`\n`', '``xml\n',
                  '<analysis>data</analysis>\n``', '`\n', '<think>also hidden</think>Done.']

        def handler(request):
            self.assertNotIn('num_predict', json.loads(request.content)['options'])
            events = [{'message': {'content': chunk}} for chunk in chunks] + [{'done': True}]
            return httpx.Response(200, text='\n'.join(map(json.dumps, events)))

        runtime = OllamaRuntime()
        runtime.prepare = AsyncMock()
        runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', transport=httpx.MockTransport(handler))
        answer = ''.join([chunk async for chunk in runtime.stream([])])
        self.assertEqual(answer, 'Code: `<think>`\n```xml\n<analysis>data</analysis>\n```\nDone.')


if __name__ == '__main__':
    unittest.main()
