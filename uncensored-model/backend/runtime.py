"""The only model adapter. No remote fallback or automatic downloads."""
import json
import asyncio
import re
from hashlib import sha256

import httpx
from pydantic import ValidationError

from .models import GenerationOptions, REGISTRY, TextModel

RUNTIME_URL = 'http://127.0.0.1:11434'
VISION_MODEL = TextModel(
    id='abyss-vision', displayName='Qwen3-VL 30B-A3B Abliterated (Q4_K_M)',
    ollamaModel='hf.co/mradermacher/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-GGUF:Q4_K_M',
)
VISION_MISSING = f'Image attachments require {VISION_MODEL.displayName} in local Ollama. Install it manually, then refresh models.'
VISION_UNSUPPORTED = f'Ollama does not report vision capability for {VISION_MODEL.displayName}. Check that the installed build includes vision support.'


def verify_vision(metadata):
    if 'vision' not in (metadata.get('capabilities') or []):
        raise RuntimeErrorLocal(VISION_UNSUPPORTED)


REASONING_TAGS = tuple(f'<{slash}{name}>' for name in ('think', 'thinking', 'analysis', 'reasoning')
                       for slash in ('', '/'))
REASONING_DELIMITER = re.compile('|'.join([*map(re.escape, REASONING_TAGS), r'`+']), re.IGNORECASE)


class AnswerText:
    """Keep only visible text, retaining just enough input to match split tags."""

    def __init__(self):
        self.pending = ''
        self.blocks = []
        self.code_ticks = 0
        self.saw_reasoning = False

    def feed(self, chunk, *, final=False):
        text = self.pending + chunk
        self.pending = ''
        answer = []
        start = 0
        for tag in REASONING_DELIMITER.finditer(text):
            if not self.blocks:
                answer.append(text[start:tag.start()])
            delimiter = tag[0].lower()
            start = tag.end()
            if delimiter.startswith('`'):
                if not self.blocks:
                    if start == len(text) and not final:
                        self.pending = delimiter
                        break
                    answer.append(delimiter)
                    if not self.code_ticks:
                        self.code_ticks = len(delimiter)
                    elif self.code_ticks == len(delimiter):
                        self.code_ticks = 0
            elif self.code_ticks:
                answer.append(tag[0])
            elif delimiter.startswith('</'):
                if self.blocks and self.blocks[-1] == delimiter[2:-1]:
                    self.blocks.pop()
            else:
                self.blocks.append(delimiter[1:-1])
                self.saw_reasoning = True
        tail = text[start:]
        marker_start = tail.rfind('<')
        if marker_start >= 0 and not self.code_ticks:
            candidate = tail[marker_start:].lower()
            if any(tag.startswith(candidate) for tag in REASONING_TAGS):
                if not final or len(candidate.lstrip('</')) >= 2:
                    self.pending = '' if final else tail[marker_start:]
                    tail = tail[:marker_start]
        if not self.blocks:
            answer.append(tail)
        return ''.join(answer)


class RuntimeErrorLocal(Exception):
    pass


class ContextLimitError(RuntimeErrorLocal):
    pass


def context_error(detail, limit):
    if 'context' in detail.lower() and any(word in detail.lower() for word in ('length', 'window', 'too long', 'exceed')):
        return ContextLimitError(
            f'This request exceeds the model’s available context window (up to {limit:,} tokens). '
            'Shorten or split the prompt, or use a model with a larger configured context. No prompt was truncated.')
    return None


def ollama_error(value, fallback):
    if isinstance(value, str) and value.strip():
        return ' '.join(value.split())[:500]
    return fallback


def ollama_http_error(response):
    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = ollama_error(body.get('error') if isinstance(body, dict) else None, 'No error details were supplied.')
    return f'Ollama {response.request.url.path} returned HTTP {response.status_code}. {detail}'


def discovered_models(tags):
    registered_tags = {model.ollamaModel for model in REGISTRY.models}
    registered_ids = {model.id for model in REGISTRY.models}
    for tag in sorted(tags - registered_tags):
        model_id = 'ollama-' + sha256(tag.encode('utf-8')).hexdigest()
        if model_id in registered_ids:
            continue
        try:
            yield TextModel(id=model_id, displayName=tag, ollamaModel=tag,
                            options=GenerationOptions(num_ctx=16384))
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

    async def vision_status(self, installed, *, available):
        detail = VISION_MISSING if available else 'Could not read installed models from local Ollama. Check that Ollama is running, then refresh models.'
        if installed:
            try:
                async with self.client() as client:
                    metadata = await self.show_local_model(client, VISION_MODEL, timeout=3)
                verify_vision(metadata)
                detail = None
            except RuntimeErrorLocal as exc:
                detail = str(exc)
            except httpx.TimeoutException:
                detail = 'Ollama timed out while checking vision capability. Refresh models to retry.'
            except (httpx.HTTPError, ValueError):
                detail = 'Could not verify the local vision model with Ollama. Refresh models and try again.'
        return {'model': VISION_MODEL.ollamaModel, 'displayName': VISION_MODEL.displayName,
                'installed': installed, 'ready': detail is None, 'detail': detail}

    async def show_local_model(self, client, selected, **request_options):
        shown = await client.post('/api/show', json={'model': selected.ollamaModel}, **request_options)
        shown.raise_for_status()
        metadata = shown.json()
        if not isinstance(metadata, dict):
            raise ValueError('Expected an Ollama model information object.')
        if metadata.get('remote_host') or metadata.get('remote_model') or not metadata.get('model_info'):
            raise RuntimeErrorLocal('Only installed local models with local weights are allowed.')
        return metadata

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
            raise RuntimeErrorLocal(VISION_MISSING if selected == VISION_MODEL else 'Selected model is not installed. Follow README setup.')
        metadata = await self.show_local_model(client, selected)
        if selected == VISION_MODEL:
            verify_vision(metadata)
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
        return metadata

    async def stream(self, messages, model_id=None, *, attachment_context=None):
        try:
            selected = VISION_MODEL if model_id == VISION_MODEL.id else REGISTRY.get(model_id)
        except ValueError:
            try:
                selected = model_from_catalog(await self.catalog(), model_id)
            except ValueError as exc:
                raise RuntimeErrorLocal('Selected model is no longer available. Refresh Models and try again.') from exc
        vision = selected == VISION_MODEL and any(message.get('images') for message in messages)
        options = selected.options.model_dump(exclude_none=True)
        if attachment_context:
            options['num_ctx'] = max(options['num_ctx'], attachment_context)
        if REGISTRY.system_instruction and REGISTRY.system_instruction.strip():
            messages = [{'role': 'system', 'content': REGISTRY.system_instruction}, *messages]
        answer_text = AnswerText()
        visible = thinking = False
        try:
            async with self.client() as client:
                metadata = await self.prepare(client, selected)
                if isinstance(metadata, dict):
                    info = metadata.get('model_info', {})
                    trained_context = info.get(f"{info.get('general.architecture')}.context_length")
                    if isinstance(trained_context, int) and trained_context > 0:
                        options['num_ctx'] = min(options['num_ctx'], trained_context)
                async with client.stream('POST', '/api/chat', json={
                    'model': selected.ollamaModel, 'messages': messages, 'stream': True,
                    'truncate': False, 'shift': False,
                    **({'think': selected.think} if selected.think is not None else {}),
                    'keep_alive': 0,
                    'options': options,
                }) as response:
                    if response.status_code != 200:
                        await response.aread()
                        detail = ollama_http_error(response)
                        raise context_error(detail, options['num_ctx']) or RuntimeErrorLocal(detail)
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError('Expected an Ollama response object.')
                        if event.get('error'):
                            detail = ollama_error(event['error'], 'No error details were supplied.')
                            raise context_error(detail, options['num_ctx']) or RuntimeErrorLocal('Ollama generation failed. ' + detail)
                        message = event.get('message', {})
                        if not isinstance(message, dict):
                            raise ValueError('Expected an Ollama message object.')
                        thinking |= bool(message.get('thinking'))
                        content = message.get('content', '')
                        if not isinstance(content, str):
                            raise ValueError('Expected text in the Ollama message content.')
                        done = bool(event.get('done'))
                        if content or done:
                            content = answer_text.feed(content, final=done)
                            if content:
                                visible |= bool(content.strip())
                                yield content
                        if done:
                            if vision and visible and event.get('done_reason') == 'length':
                                raise RuntimeErrorLocal('Ollama reached its output token limit. The answer shown is incomplete.')
                            if not visible:
                                if event.get('done_reason') == 'length':
                                    raise RuntimeErrorLocal('Ollama reached its output token limit before returning an answer. Retry with a more focused question.')
                                if thinking or answer_text.saw_reasoning:
                                    raise RuntimeErrorLocal('Ollama returned thinking tokens but no answer. Retry with a more focused question.')
                                raise RuntimeErrorLocal('Ollama finished without returning an answer. You can retry this response.')
                            return
                    raise RuntimeErrorLocal('The model stream ended early. You can retry this response.')
        except httpx.HTTPStatusError as exc:
            raise RuntimeErrorLocal(ollama_http_error(exc.response)) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeErrorLocal('Ollama timed out while loading the model or generating an answer. You can retry this response.') from exc
        except httpx.RequestError as exc:
            raise RuntimeErrorLocal('Connection to Ollama failed. Check that it is running locally.') from exc
        except ValueError as exc:
            raise RuntimeErrorLocal('Ollama returned an invalid response. You can retry this response.') from exc


def context_messages(messages, budget, *, omitted=None):
    # Budget older turns conservatively. Always include the latest user turn
    # intact and let Ollama enforce its actual model-specific token limit.
    turns = []
    for message in messages:
        if message['role'] == 'user':
            turns.append([{'role': 'user', 'content': message['content']}])
        elif turns and message['status'] == 'complete' and message['content'].strip():
            turns[-1].append({'role': 'assistant', 'content': message['content']})
    selected, size = [], 0
    for index, turn in enumerate(reversed(turns)):
        cost = sum(len(m['content'].encode('utf-8')) + 32 for m in turn)
        if index and size + cost > budget:
            if omitted is not None:
                omitted.append(len(turns) - index)
            break
        selected = turn + selected
        size += cost
    return selected
