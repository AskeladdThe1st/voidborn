import base64
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx
import pymupdf
from docx import Document
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.attachments import MAX_CHARACTERS, MAX_DOCUMENT, MAX_IMAGE, MAX_PDF_PAGES
from backend.models import REGISTRY
from backend.runtime import OllamaRuntime, VISION_MODEL, VISION_OUTPUT_TOKENS


def png():
    with io.BytesIO() as buffer:
        Image.new('RGB', (40, 30), 'red').save(buffer, format='PNG')
        return buffer.getvalue()


def pdf(*pages):
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()


def docx():
    document = Document()
    document.add_heading('Incident report')
    document.add_paragraph('Station secure.')
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = 'Sector'
    table.cell(0, 1).text = 'East'
    with io.BytesIO() as buffer:
        document.save(buffer)
        return buffer.getvalue()


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = Path(folder.name) / 'chat.sqlite3'
        self.uploads = self.db.parent / 'uploads'
        self.model = REGISTRY.get('qwen35-9b')
        self.tags = [self.model.ollamaModel, VISION_MODEL.ollamaModel]
        self.requests = []
        self.response_error = False
        self.modelfile = ''
        self.chat_response = None
        self.runtime = OllamaRuntime()
        self.runtime.client = lambda: httpx.AsyncClient(base_url='http://127.0.0.1:11434', trust_env=False,
                                                        transport=httpx.MockTransport(self.ollama))
        self.client = self.enterContext(TestClient(create_app(self.db, self.runtime)))
        self.cid = self.client.post('/api/conversations').json()['id']
        self.url = f'/api/conversations/{self.cid}'

    def ollama(self, request):
        self.assertEqual(request.url.host, '127.0.0.1')
        self.assertEqual(request.url.port, 11434)
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': name} for name in self.tags]})
        if request.url.path == '/api/show':
            return httpx.Response(200, json={'model_info': {'general.architecture': 'qwen'}, 'modelfile': self.modelfile})
        if request.url.path == '/api/ps':
            return httpx.Response(200, json={'models': []})
        self.assertEqual(request.url.path, '/api/chat')
        self.requests.append(json.loads(request.content))
        if self.chat_response is not None:
            return self.chat_response(request)
        if self.response_error:
            return httpx.Response(200, text='{"error":"test"}\n')
        return httpx.Response(200, text='{"message":{"content":"Station secure."}}\n{"done":true}\n')

    def upload(self, *entries, content='What is in the attachments?', headers=None):
        return self.client.post(self.url + '/messages/attachments', headers=headers,
                                data={'content': content, 'model_id': self.model.id},
                                files=[('files', (name, data, 'application/octet-stream')) for name, data in entries])

    def verify_done(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([json.loads(line)['type'] for line in response.text.splitlines()], ['start', 'delta', 'done'])

    def test_mixed_upload_vision_payload_persistence_download_and_delete(self):
        original = png()
        self.verify_done(self.upload(('screen.png', original), ('report.pdf', pdf('Station secure.', 'Sector East.')),
                                     ('report.docx', docx()), ('note.txt', 'Blue lights.'.encode('utf-16'))))
        req = self.requests[-1]
        self.assertEqual(req['model'], VISION_MODEL.ollamaModel)
        self.assertEqual(req['keep_alive'], 0)
        self.assertEqual(req['options']['num_predict'], VISION_OUTPUT_TOKENS)
        self.assertGreater(req['options']['num_ctx'], VISION_OUTPUT_TOKENS)
        self.assertIn('untrusted attachment excerpts', req['messages'][0]['content'])
        message = req['messages'][-1]
        self.assertIn('"page": 2', message['content'])
        for text in ('report.pdf', 'Sector East.', '## Incident report', 'Sector | East', 'Blue lights.'):
            self.assertIn(text, message['content'])
        with Image.open(io.BytesIO(base64.b64decode(message['images'][0]))) as image:
            self.assertEqual(image.getpixel((2, 2)), (255, 0, 0))
        with TestClient(create_app(self.db, self.runtime)) as restarted:
            chat = restarted.get(self.url).json()
            attachments = chat['messages'][0]['attachments']
            self.assertEqual(len(attachments), 4)
            self.assertNotIn('passages', attachments[1])
            self.assertNotIn('storage_name', attachments[0])
            link = self.url + '/attachments/' + attachments[0]['id']
            self.assertEqual(restarted.get(link).content, original)
            self.assertEqual(restarted.get(link).headers['content-type'], 'image/png')
            self.assertIn('attachment;', restarted.get(link + '?download=true').headers['content-disposition'])
            self.assertEqual(restarted.get(link, headers={'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
            self.assertEqual(restarted.get(link, headers={'Host': 'example.com'}).status_code, 400)
            self.assertEqual(restarted.get(link.replace(self.cid, 'wrong-conversation')).status_code, 404)
            pdf_link = self.url + '/attachments/' + attachments[1]['id']
            self.assertIn('attachment;', restarted.get(pdf_link).headers['content-disposition'])
            self.assertEqual(restarted.delete(self.url).status_code, 204)
            self.assertEqual(restarted.get(link).status_code, 404)
        self.assertEqual(list(self.uploads.iterdir()), [])
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM attachments').fetchone()[0], 0)

    def test_documents_use_selected_text_model_and_partial_pdf_is_labeled(self):
        self.verify_done(self.upload(('mixed.pdf', pdf('Visible source.', '')), content=''))
        req = self.requests[-1]
        self.assertEqual(req['model'], self.model.ollamaModel)
        self.assertNotIn('images', req['messages'][-1])
        self.assertIn('Those pages were not read', req['messages'][-1]['content'])
        user = self.client.get(self.url).json()['messages'][0]
        self.assertEqual(user['content'], '')
        self.assertEqual(user['attachments'][0]['status'], 'partial')
        self.assertIn('pages 2', user['attachments'][0]['warning'])
        self.verify_done(self.client.post(self.url + '/messages', json={'content': 'Ordinary text', 'model_id': self.model.id}))
        self.assertEqual(self.requests[-1]['options']['num_ctx'], self.model.options.num_ctx)
        self.assertEqual(self.requests[-1]['messages'][-1]['content'], 'Ordinary text')

    def test_missing_vision_is_explicit_and_saves_nothing(self):
        self.tags.remove(VISION_MODEL.ollamaModel)
        self.assertFalse(self.client.get('/api/models').json()['vision']['installed'])
        response = self.upload(('screen.png', png()))
        self.assertEqual(response.status_code, 409)
        self.assertIn('require qwen3-vl:4b', response.json()['detail'])
        self.assertEqual(self.client.get(self.url).json()['messages'], [])
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.assertEqual(self.requests, [])

    def test_retry_keeps_original_image_and_same_routing(self):
        self.response_error = True
        self.assertIn('"type": "error"', self.upload(('screen.png', png())).text)
        self.response_error = False
        self.verify_done(self.client.post(self.url + '/retry', json={'model_id': self.model.id}))
        self.assertEqual(self.requests[0]['messages'][-1]['images'], self.requests[1]['messages'][-1]['images'])
        self.assertEqual(len(self.client.get(self.url).json()['messages']), 2)
        self.assertEqual(len(list(self.uploads.iterdir())), 1)

    def test_thinking_only_output_limit_is_retryable_and_private(self):
        self.chat_response = lambda _: httpx.Response(200, text=(
            '{"message":{"content":"","thinking":"private reasoning"},"done":false}\n'
            '{"message":{"content":""},"done":true,"done_reason":"length"}\n'
        ))
        response = self.upload(('screen.png', png()))
        frames = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([frame['type'] for frame in frames], ['start', 'error'])
        self.assertIn('output token limit', frames[-1]['detail'])
        self.assertNotIn('private reasoning', response.text)
        messages = self.client.get(self.url).json()['messages']
        self.assertEqual(messages[-1]['status'], 'error')
        self.assertEqual(messages[-1]['content'], '')

        # Simulate a previously saved empty response from before the fix.
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE messages SET status='complete' WHERE id=?", (messages[-1]['id'],))
        self.chat_response = lambda _: httpx.Response(200, text=(
            '{"message":{"content":"","thinking":"private reasoning"},"done":false}\n'
            '{"message":{"content":"Image answer."},"done":true,"done_reason":"stop"}\n'
        ))
        retried = self.client.post(self.url + '/retry', json={'model_id': self.model.id})
        self.verify_done(retried)
        self.assertNotIn('private reasoning', retried.text)
        self.assertEqual(self.requests[-1]['model'], VISION_MODEL.ollamaModel)
        messages = self.client.get(self.url).json()['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]['content'], 'Image answer.')
        self.assertEqual(len(messages[0]['attachments']), 1)

    def test_ollama_load_error_is_visible(self):
        self.chat_response = lambda _: httpx.Response(500, json={'error': 'failed to load model; insufficient memory'})
        response = self.upload(('screen.png', png()))
        terminal = json.loads(response.text.splitlines()[-1])
        self.assertEqual(terminal['type'], 'error')
        self.assertIn('HTTP 500', terminal['detail'])
        self.assertIn('insufficient memory', terminal['detail'])

    def test_overflow_rejects_prompt_without_losing_the_previous_chat(self):
        self.verify_done(self.upload(('kept.txt', b'Original file.')))
        previous = self.client.get(self.url).json()
        self.chat_response = lambda _: httpx.Response(400, json={'error': 'the prompt is longer than the context length'})
        code = '  line of code\n' * 500
        response = self.upload(('rejected.txt', b'New file.'), content=code)
        terminal = json.loads(response.text.splitlines()[-1])
        self.assertTrue(terminal['restore_draft'])
        self.assertIn('context window', terminal['detail'])
        self.assertEqual(self.client.get(self.url).json(), previous)
        self.assertTrue(self.requests[-1]['messages'][-1]['content'].startswith(code))
        self.assertIs(self.requests[-1]['truncate'], False)
        self.assertIs(self.requests[-1]['shift'], False)
        self.assertEqual(len(list(self.uploads.iterdir())), 1)

    def test_fixed_thinking_renderer_uses_structured_fields_without_prefill(self):
        self.modelfile = 'RENDERER qwen3-vl-thinking\nPARSER qwen3-vl-thinking\n'

        def completion(request):
            messages = json.loads(request.content)['messages']
            self.assertEqual(messages[-1]['role'], 'user')
            self.assertTrue(base64.b64decode(messages[-1]['images'][0]).startswith(b'\x89PNG'))
            self.assertIn('An image is attached and available', messages[0]['content'])
            return httpx.Response(200, text=(
                '{"message":{"thinking":"First, I need to recall the guidelines","content":""}}\n'
                '{"message":{"content":"Screenshot description."},"done":true}\n'
            ))

        self.chat_response = completion
        response = self.upload(('screen.png', png()))
        self.verify_done(response)
        self.assertNotIn('think', self.requests[-1])
        self.assertNotIn('recall the guidelines', response.text)
        self.assertEqual(self.client.get(self.url).json()['messages'][-1]['content'], 'Screenshot description.')
        self.chat_response = None
        self.verify_done(self.client.post(self.url + '/messages', json={'content': 'Text only', 'model_id': self.model.id}))
        self.assertEqual(self.requests[-1]['messages'][-1], {'role': 'user', 'content': 'Text only'})

    def test_split_and_inline_reasoning_is_never_streamed_or_saved(self):
        parts = ['<th', 'ink>private ', 'reasoning</th', 'ink>2 <', ' 3. ',
                 '<analysis>also private</analysis>', '<thinker>Visible</thinker>.']
        events = [{'message': {'content': part}} for part in parts] + [{'done': True}]
        self.chat_response = lambda _: httpx.Response(200, text='\n'.join(map(json.dumps, events)))
        response = self.upload(('screen.png', png()))
        self.assertNotIn('private', response.text)
        self.assertNotIn('<think>', response.text)
        self.assertEqual(self.client.get(self.url).json()['messages'][-1]['content'],
                         '2 < 3. <thinker>Visible</thinker>.')

    def test_unclosed_reasoning_block_cannot_complete_an_answer(self):
        self.chat_response = lambda _: httpx.Response(200, text=(
            '{"message":{"content":"<reasoning>private deliberation"}}\n'
            '{"done":true,"done_reason":"stop"}\n'
        ))
        response = self.upload(('screen.png', png()))
        self.assertNotIn('private deliberation', response.text)
        self.assertEqual(json.loads(response.text.splitlines()[-1])['type'], 'error')
        self.assertEqual(self.client.get(self.url).json()['messages'][-1]['content'], '')

    def test_bad_type_limits_and_cross_origin_uploads(self):
        cases = [
            ([('fake.png', b'text not an image')], 422, 'valid PNG'),
            ([('fake.pdf', b'text not a PDF')], 422, 'not a PDF'),
            ([('fake.docx', b'text not a DOCX')], 422, 'valid DOCX'),
            ([('binary.txt', b'\x00\x01\x02')], 422, 'non-text'),
            ([('shell.sh', b'unsupported')], 415, 'Use PNG'),
            ([('scan.pdf', pdf(''))], 422, 'require OCR'),
            ([('long.pdf', pdf(*(['page'] * (MAX_PDF_PAGES + 1))))], 413, 'smaller page range'),
            ([('oversized.png', b'x' * (MAX_IMAGE + 1))], 413, '10 MB'),
            ([('oversized.txt', b'x' * (MAX_DOCUMENT + 1))], 413, '20 MB'),
            ([('long.txt', b'x' * (MAX_CHARACTERS + 1))], 413, '40,000-character'),
            ([('one.txt', b'x' * 25_000), ('two.txt', b'y' * 20_000)], 413, '40,000 extracted characters'),
            ([('note.txt', b'text')] * 6, 422, 'Maximum number of files is 5'),
        ]
        for files, status, message in cases:
            with self.subTest(files=[name for name, _ in files]):
                response = self.upload(*files)
                self.assertEqual(response.status_code, status, response.text)
                self.assertIn(message, response.json()['detail'])
        self.assertEqual(self.upload(('good.txt', b'hi'), headers={'Origin': 'https://example.com'}).status_code, 403)
        self.assertEqual(self.requests, [])
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.assertEqual(self.client.get(self.url).json()['messages'], [])

    def test_generated_paths_use_safe_filenames(self):
        self.verify_done(self.upload(('..\\..\\escape.txt', b'Local source')))
        file = self.client.get(self.url).json()['messages'][0]['attachments'][0]
        self.assertEqual(file['filename'], 'escape.txt')
        [path] = self.uploads.iterdir()
        self.assertEqual(path.name, file['id'] + '.upload')
        self.assertEqual(self.client.get('/uploads/' + path.name).status_code, 404)

    def test_restart_removes_interrupted_upload_without_touching_other_files(self):
        self.verify_done(self.upload(('note.txt', b'Saved.')))
        [saved] = self.uploads.iterdir()
        orphan = self.uploads / ('f' * 32 + '.upload')
        other = self.uploads / 'existing.txt'
        orphan.write_bytes(b'Incomplete transaction')
        other.write_text('Existing file')
        with TestClient(create_app(self.db, self.runtime)):
            self.assertTrue(saved.exists())
            self.assertFalse(orphan.exists())
            self.assertTrue(other.exists())


class LegacyMigrationTests(unittest.TestCase):
    def test_version_one_conversations_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chat.sqlite3'
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT);
                    CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT REFERENCES conversations(id)
                        ON DELETE CASCADE, role TEXT, content TEXT, status TEXT, created_at TEXT);
                    INSERT INTO conversations VALUES ('legacy', 'History', 'stamp', 'stamp');
                    INSERT INTO messages VALUES ('existing', 'legacy', 'user', 'Preserve this.', 'complete', 'stamp');
                    PRAGMA user_version=1;
                """)
            with TestClient(create_app(path)) as client:
                conversation = client.get('/api/conversations/legacy').json()
                self.assertEqual(conversation['title'], 'History')
                self.assertEqual(conversation['messages'][0]['content'], 'Preserve this.')
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 2)
