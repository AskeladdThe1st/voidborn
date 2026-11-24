import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.dictation import MAX_BYTES, SAMPLE_RATE


def recording(*, rate=SAMPLE_RATE, frames=8_000, sample=1000):
    output = io.BytesIO()
    with wave.open(output, 'wb') as stream:
        stream.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        stream.writeframes(struct.pack('<h', sample) * frames)
    return output.getvalue()


class DictationTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.client = self.enterContext(TestClient(create_app(self.folder / 'chat.sqlite3')))

    def upload(self, body, mime='audio/wav', **headers):
        return self.client.post('/api/transcribe', content=body, headers={'Content-Type': mime, **headers})

    def test_returns_only_text_without_storing_audio_or_chat_messages(self):
        before = set(self.folder.rglob('*'))
        with patch('backend.dictation.transcribe', return_value='A local transcript.') as whisper:
            response = self.upload(recording())
        self.assertEqual(response.json(), {'text': 'A local transcript.'})
        self.assertEqual(response.headers['Permissions-Policy'], 'microphone=(self)')
        self.assertEqual(whisper.call_args.args[0], struct.pack('<h', 1000) * 8_000)
        self.assertEqual(set(self.folder.rglob('*')), before)
        self.assertEqual(self.client.get('/api/conversations').json(), [])

    def test_rejects_wrong_format_empty_audio_and_oversized_body(self):
        with patch('backend.dictation.transcribe') as whisper:
            self.assertEqual(self.upload(recording(), 'audio/webm').status_code, 415)
            self.assertEqual(self.upload(recording(rate=8000)).status_code, 415)
            self.assertEqual(self.upload(recording()[:-2]).status_code, 422)
            self.assertEqual(self.upload(recording(sample=0)).status_code, 422)
            self.assertEqual(self.upload(recording(frames=100)).status_code, 422)
            self.assertEqual(self.upload(b'x' * (MAX_BYTES + 1)).status_code, 413)
            whisper.assert_not_called()

    def test_local_only_and_missing_weights(self):
        self.assertEqual(self.upload(recording(), Origin='https://example.com').status_code, 403)
        self.assertEqual(self.upload(recording(), Host='example.com').status_code, 400)
        with patch('backend.dictation.MODEL_PATH', self.folder / 'no-model'):
            response = self.upload(recording())
        self.assertEqual(response.status_code, 503)
        self.assertIn('local Whisper Small model is not installed', response.json()['detail'])
