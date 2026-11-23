"""Real Ollama comparison against a temporary database; leaves user history alone."""
import json
import tempfile
import threading
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app


def main():
    stop = threading.Event()
    counts = []

    def monitor():
        with httpx.Client(base_url='http://127.0.0.1:11434', trust_env=False, timeout=5) as runtime:
            while not stop.is_set():
                counts.append(len(runtime.get('/api/ps').json().get('models', [])))
                stop.wait(0.2)

    watcher = threading.Thread(target=monitor)
    watcher.start()
    try:
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / 'chat.sqlite3')) as client:
                catalog = client.get('/api/models').json()
                assert all(m['installed'] for m in catalog['models']), catalog
                response = client.post('/api/compare', json={'content': 'What is 2 + 2? Answer in a few words.'})
                response.raise_for_status()
                events = [json.loads(line) for line in response.text.splitlines()]
                failures = [e for e in events if e['type'] == 'model_error']
                assert not failures, failures
                results = [e for e in events if e['type'] == 'result']
                assert len(results) == len(catalog['models'])
                for result in results:
                    answer = ''.join(e['content'] for e in events if e['type'] == 'delta' and e['model_id'] == result['model_id'])
                    assert answer.strip()
                    print(json.dumps({**result, 'response': answer}, ensure_ascii=True), flush=True)
                assert client.get('/api/conversations').json() == []
                assert events[-1]['type'] == 'done'
    finally:
        stop.set()
        watcher.join()
    assert counts and max(counts) <= 1, counts
    print(f'PASS: sequential comparison; maximum loaded models observed: {max(counts)}.', flush=True)


if __name__ == '__main__':
    main()
