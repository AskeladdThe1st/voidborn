"""Optional real-model smoke test. Requires the app and Ollama running."""
import json
import time
import httpx


with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=300) as client:
    page = client.get('/')
    page.raise_for_status()
    assert 'Abyss' in page.text
    assert "connect-src 'self'" in page.headers['content-security-policy']
    status = client.get('/api/status').json()
    assert status['ready'], status
    cid = client.post('/api/conversations').json()['id']
    path = '/api/conversations/' + cid
    try:
        start = time.monotonic()
        deltas, answer, first = 0, '', None
        with client.stream('POST', path + '/messages', json={'content': 'Reply with a brief greeting in one sentence.'}) as response:
            response.raise_for_status()
            done = False
            for line in response.iter_lines():
                event = json.loads(line)
                if event['type'] == 'error':
                    raise RuntimeError(event['detail'])
                if event['type'] == 'delta':
                    deltas += 1
                    answer += event['content']
                    if first is None:
                        first = time.monotonic() - start
                        print(f'First token: {first:.2f}s', flush=True)
                if event['type'] == 'done':
                    done = True
        assert done and answer and deltas > 1
        saved = client.get(path).json()['messages']
        assert saved[-1]['content'] == answer and saved[-1]['status'] == 'complete'
        client.patch(path, json={'title': 'Smoke test renamed'}).raise_for_status()
        assert client.get(path).json()['title'] == 'Smoke test renamed'
        print(f'PASS: {deltas} streamed chunks, persisted response, rename. Response: {answer}', flush=True)
    finally:
        client.delete(path).raise_for_status()
