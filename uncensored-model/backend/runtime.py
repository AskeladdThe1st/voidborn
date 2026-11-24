"""The only model adapter. No remote fallback or automatic downloads."""
import json
import asyncio
from hashlib import sha256

import httpx
from pydantic import ValidationError

from .models import REGISTRY, TextModel

RUNTIME_URL = 'http://127.0.0.1:11434'


class RuntimeErrorLocal(Exception):
    pass


def discovered_models(tags):
    registered_tags = {model.ollamaModel for model in REGISTRY.models}
    registered_ids = {model.id for model in REGISTRY.models}
    for tag in sorted(tags - registered_tags):
        model_id = 'ollama-' + sha256(tag.encode('utf-8')).hexdigest()
        if model_id in registered_ids:
            continue
        try:
            yield TextModel(id=model_id, displayName=tag, ollamaModel=tag)
        except ValidationError:
            continue


def model_from_catalog(catalog, model_id):
    for entry in catalog['models']:
        if entry['id'] == model_id:
            return TextModel.model_validate({key: value for key, value in entry.items() if key != 'installed'})
    raise ValueError('Unknown model ID')


class OllamaRuntime:
    def client(self):
        return httpx.AsyncClient(base_url=RUNTIME_URL, trust_env=False,
                                 follow_redirects=False, timeout=httpx.Timeout(300, connect=3))

    async def catalog(self):
        try:
            async with self.client() as client:
                response = await client.get('/api/tags', timeout=3)
                response.raise_for_status()
                tags = {m['name'] for m in response.json().get('models', [])
                        if isinstance(m.get('name'), str) and not (m.get('remote_host') or m.get('remote_model'))}
            return {'default_model': REGISTRY.default_model, 'available': True,
                    'models': [{**m.model_dump(), 'installed': m.ollamaModel in tags}
                               for m in [*REGISTRY.models, *discovered_models(tags)]]}
        except (httpx.HTTPError, ValueError):
            return {'default_model': REGISTRY.default_model, 'available': False,
                    'models': [{**m.model_dump(), 'installed': False} for m in REGISTRY.models]}

    async def status(self, model_id=None):
        catalog = await self.catalog()
        selected = model_from_catalog(catalog, model_id or REGISTRY.default_model)
        ready = next(m['installed'] for m in catalog['models'] if m['id'] == selected.id)
        return {'ready': ready, 'model': selected.ollamaModel,
                'detail': 'Ready' if ready else ('Model not installed. Follow README setup.' if catalog['available'] else 'Ollama is unavailable at 127.0.0.1:11434.')}

    async def prepare(self, client, selected):
        # Verify installation before generation; never implicitly pull a model.
        tags = await client.get('/api/tags')
        tags.raise_for_status()
        if not any(m.get('name') == selected.ollamaModel for m in tags.json().get('models', [])):
            raise RuntimeErrorLocal('Selected model is not installed. Follow README setup.')
        shown = await client.post('/api/show', json={'model': selected.ollamaModel})
        shown.raise_for_status()
        metadata = shown.json()
        if metadata.get('remote_host') or metadata.get('remote_model') or not metadata.get('model_info'):
            raise RuntimeErrorLocal('Only installed local models with local weights are allowed.')
        running = await client.get('/api/ps')
        running.raise_for_status()
        for loaded in running.json().get('models', []):
            tag = loaded.get('name') or loaded.get('model')
            if tag != selected.ollamaModel:
                unloaded = await client.post('/api/generate', json={'model': tag, 'keep_alive': 0, 'stream': False})
                unloaded.raise_for_status()
        # Fail closed if the runtime has not released another model yet.
        for attempt in range(30):
            running = await client.get('/api/ps')
            running.raise_for_status()
            if not any((m.get('name') or m.get('model')) != selected.ollamaModel for m in running.json().get('models', [])):
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeErrorLocal('Another model is still loaded. Wait briefly and retry.')

    async def stream(self, messages, model_id=None):
        try:
            selected = REGISTRY.get(model_id)
        except ValueError:
            try:
                selected = model_from_catalog(await self.catalog(), model_id)
            except ValueError as exc:
                raise RuntimeErrorLocal('Selected model is no longer available. Refresh Models and try again.') from exc
        instruction = REGISTRY.system_instruction + ('\n/no_think' if selected.no_think else '')
        messages = [{'role': 'system', 'content': instruction}, *messages]
        prefix, at_start = '', selected.no_think
        try:
            async with self.client() as client:
                await self.prepare(client, selected)
                async with client.stream('POST', '/api/chat', json={
                    'model': selected.ollamaModel, 'messages': messages, 'stream': True,
                    **({'think': selected.think} if selected.think is not None else
                       ({'think': False} if selected.no_think else {})),
                    'keep_alive': 0,
                    'options': selected.options.model_dump(),
                }) as response:
                    if response.status_code != 200:
                        raise RuntimeErrorLocal('Ollama could not run the model. Check its installation and available memory.')
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get('error'):
                            raise RuntimeErrorLocal('Ollama reported a generation error. Check the local runtime.')
                        content = event.get('message', {}).get('content', '')
                        if content:
                            if at_start:
                                prefix += content
                                candidate = prefix.lstrip()
                                marker = '</think>'
                                if candidate.startswith(marker):
                                    content = candidate[len(marker):].lstrip('\r\n')
                                    at_start = False
                                elif marker.startswith(candidate):
                                    content = ''
                                else:
                                    content, at_start = prefix, False
                            if content:
                                yield content
                        if event.get('done'):
                            if at_start and prefix.strip():
                                yield prefix
                            return
                    raise RuntimeErrorLocal('The model stream ended early. You can retry this response.')
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeErrorLocal('Connection to Ollama failed or timed out. Check that it is running locally.') from exc


def context_messages(messages):
    # Conservative UTF-8 byte budget; full history stays in SQLite.
    turns = []
    for message in messages:
        if message['role'] == 'user':
            turns.append([{'role': 'user', 'content': message['content']}])
        elif turns and message['status'] == 'complete':
            turns[-1].append({'role': 'assistant', 'content': message['content']})
    selected, size = [], 0
    for turn in reversed(turns):
        cost = sum(len(m['content'].encode('utf-8')) + 32 for m in turn)
        if size + cost > 2800:
            break
        selected = turn + selected
        size += cost
    return selected
