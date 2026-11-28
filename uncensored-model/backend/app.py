import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .attachments import AttachmentError, attach_sources, parse_upload, remove_orphans, remove_upload, save_uploads, stored_path
from .dictation import dictation_router
from .runtime import ContextLimitError, OllamaRuntime, RuntimeErrorLocal, VISION_MODEL, context_messages, model_from_catalog
from .storage import Store
from .models import MAX_PROMPT_BYTES, REGISTRY

ROOT = Path(__file__).resolve().parent.parent


class Title(BaseModel):
    title: str = Field(min_length=1, max_length=100)

    @field_validator('title')
    @classmethod
    def clean(cls, value):
        if not value.strip():
            raise ValueError('Title cannot be blank')
        return value.strip()


class Prompt(BaseModel):
    content: str = Field(min_length=1)
    model_id: str | None = None

    @field_validator('content')
    @classmethod
    def clean(cls, value):
        if not value.strip():
            raise ValueError('Message cannot be blank.')
        check_prompt_size(value)
        return value


def check_prompt_size(content):
    if len(content.encode('utf-8')) > MAX_PROMPT_BYTES:
        raise HTTPException(413, 'The message exceeds the 128 KiB upload limit. Split it into smaller messages.')


class ModelChoice(BaseModel):
    model_id: str | None = None


def create_app(db_path=None, runtime=None):
    store = Store(Path(db_path) if db_path else ROOT / 'data' / 'chat.sqlite3')
    uploads_dir = store.path.parent / 'uploads'
    model = runtime or OllamaRuntime()
    generation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        store.initialize()
        uploads_dir.mkdir(mode=0o700, exist_ok=True)
        with store.connect() as db:
            referenced = {row['storage_name'] for row in db.execute('SELECT storage_name FROM attachments')}
        remove_orphans(uploads_dir, referenced)
        yield

    app = FastAPI(title='Abyss', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if origin and origin not in {'http://127.0.0.1:8000', 'http://localhost:8000'}:
            return JSONResponse({'detail': 'Cross-origin requests are disabled'}, status_code=403)
        if request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'detail': 'Cross-site requests are disabled'}, status_code=403)
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = "default-src 'self'; connect-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Permissions-Policy'] = 'microphone=(self)'
        response.headers['Cache-Control'] = 'no-store'
        return response

    def now():
        return datetime.now(timezone.utc).isoformat()

    def require_conversation(cid):
        with store.connect() as db:
            row = db.execute('SELECT * FROM conversations WHERE id=?', (cid,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Conversation not found')
        return dict(row)

    def require_idle():
        if generation_lock.locked():
            raise HTTPException(409, 'Wait for the current response to finish.')

    @app.get('/api/status')
    async def status(model_id: str | None = None):
        try:
            return await model.status(model_id or REGISTRY.default_model)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    async def choose(model_id):
        try:
            return REGISTRY.get(model_id)
        except ValueError:
            try:
                return model_from_catalog(await model.catalog(), model_id)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get('/api/models')
    async def models():
        catalog = await model.catalog()
        vision = next((m for m in catalog['models'] if m['ollamaModel'] == VISION_MODEL.ollamaModel), None)
        vision_status = await model.vision_status(bool(vision and vision['installed']), available=catalog['available'])
        return {**catalog, 'vision': vision_status,
                'models': [m for m in catalog['models'] if m['ollamaModel'] != VISION_MODEL.ollamaModel]}

    @app.post('/api/compare')
    async def compare(body: Prompt):
        require_idle()
        await generation_lock.acquire()
        try:
            catalog = await model.catalog()
            installed = [m for m in catalog['models'] if m['installed'] and m['ollamaModel'] != VISION_MODEL.ollamaModel]
            if not installed:
                raise HTTPException(409, 'No registered models are installed or Ollama is unavailable.')
        except BaseException:
            generation_lock.release()
            raise

        async def events():
            def event(kind, **data):
                return json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n'
            try:
                for entry in installed:
                    started, first_token = time.perf_counter(), None
                    yield event('start', model_id=entry['id'], label=entry['displayName'])
                    try:
                        # Identical fresh prompt; no conversation history for either model.
                        async for token in model.stream([{'role': 'user', 'content': body.content}], entry['id']):
                            if first_token is None:
                                first_token = time.perf_counter() - started
                            yield event('delta', model_id=entry['id'], content=token)
                        yield event('result', model_id=entry['id'], seconds=time.perf_counter() - started,
                                    first_token_seconds=first_token)
                    except Exception as exc:
                        detail = str(exc) if isinstance(exc, RuntimeErrorLocal) else 'Generation failed.'
                        yield event('model_error', model_id=entry['id'], detail=detail,
                                    seconds=time.perf_counter() - started)
                yield event('done')
            finally:
                generation_lock.release()
        return StreamingResponse(events(), media_type='application/x-ndjson')

    @app.get('/api/conversations')
    async def conversations():
        with store.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM conversations ORDER BY updated_at DESC, rowid DESC')]

    @app.post('/api/conversations', status_code=201)
    async def create():
        cid, stamp = str(uuid4()), now()
        with store.connect() as db:
            db.execute('INSERT INTO conversations VALUES (?,?,?,?)', (cid, 'New conversation', stamp, stamp))
        return require_conversation(cid)

    @app.get('/api/conversations/{cid}')
    async def get(cid: str):
        return {**require_conversation(cid), 'messages': store.messages(cid)}

    @app.get('/api/conversations/{cid}/attachments/{fid}')
    async def attachment(cid: str, fid: str, download: bool = False):
        with store.connect() as db:
            entry = db.execute('SELECT * FROM attachments WHERE conversation_id=? AND id=?', (cid, fid)).fetchone()
        if not entry:
            raise HTTPException(404, 'Attachment not found.')
        try:
            path = stored_path(uploads_dir, entry['storage_name'])
        except AttachmentError as exc:
            raise HTTPException(404, 'Attachment not found.') from exc
        if not path.is_file():
            raise HTTPException(404, 'Attachment file is missing.')
        inline = entry['mime'].startswith('image/') and not download
        return FileResponse(path, media_type=entry['mime'] if inline else 'application/octet-stream',
                            filename=entry['filename'], content_disposition_type='inline' if inline else 'attachment')

    @app.patch('/api/conversations/{cid}')
    async def rename(cid: str, body: Title):
        require_conversation(cid)
        with store.connect() as db:
            db.execute('UPDATE conversations SET title=?, updated_at=? WHERE id=?', (body.title, now(), cid))
        return require_conversation(cid)

    @app.delete('/api/conversations/{cid}', status_code=204)
    async def delete(cid: str):
        require_idle()
        require_conversation(cid)
        with store.connect() as db:
            names = [row['storage_name'] for row in db.execute(
                'SELECT storage_name FROM attachments WHERE conversation_id=?', (cid,))]
            db.execute('DELETE FROM conversations WHERE id=?', (cid,))
        for name in names:
            remove_upload(uploads_dir, name)

    async def generate(cid, content=None, model_id=None, uploads=()):
        require_idle()
        await generation_lock.acquire()
        new_files = []
        try:
            return await prepare_generation(cid, content, model_id, uploads, new_files)
        except BaseException:
            for file in new_files:
                remove_upload(uploads_dir, file.storage_name)
            generation_lock.release()
            raise

    async def prepare_generation(cid, content, model_id, uploads, new_files):
        conversation = require_conversation(cid)
        history = store.messages(cid)
        if content is None:
            if (not history or history[-1]['role'] != 'assistant'
                    or (history[-1]['status'] == 'complete' and history[-1]['content'].strip())):
                raise HTTPException(409, 'There is no interrupted response to retry.')
        elif history and history[-1]['status'] != 'complete':
            raise HTTPException(409, 'Retry the interrupted response before sending another message.')

        files = []
        if uploads:
            new_files.extend(await run_in_threadpool(save_uploads, uploads_dir, uploads))
            files = new_files
        elif content is None:
            previous = next(message for message in reversed(history) if message['role'] == 'user')
            files = store.attachment_sources(cid, previous['id'])

        if any(file.mime.startswith('image/') for file in files):
            vision = (await models())['vision']
            if not vision['ready']:
                raise HTTPException(409, vision['detail'])
            selected = VISION_MODEL
        else:
            selected = await choose(model_id)
        omitted = []
        context = context_messages(history[:-1] if content is None else [
            *history, {'role': 'user', 'content': content},
        ], selected.options.num_ctx - (selected.options.num_predict or 1024) - 1024, omitted=omitted)
        context_size = None
        if files:
            context_size = await run_in_threadpool(attach_sources, uploads_dir, context[-1], files, context)

        aid, stamp = str(uuid4()), now()
        mid = None
        with store.connect() as db:
            if content is None:
                db.execute('DELETE FROM messages WHERE id=?', (history[-1]['id'],))
            else:
                mid = str(uuid4())
                db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)', (mid, cid, 'user', content, 'complete', stamp))
                store.save_attachments(db, cid, mid, new_files)
                if not history and conversation['title'] == 'New conversation':
                    title = content.strip() or new_files[0].filename
                    db.execute('UPDATE conversations SET title=? WHERE id=?', (title[:50], cid))
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)', (aid, cid, 'assistant', '', 'streaming', stamp))
            db.execute('UPDATE conversations SET updated_at=? WHERE id=?', (stamp, cid))

        async def events():
            answer, state = '', 'interrupted'
            def event(kind, **data):
                return json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n'
            try:
                attachments = [{key: getattr(file, key) for key in ('id', 'filename', 'mime', 'size', 'status', 'warning')}
                               for file in new_files]
                yield event('start', message_id=aid, model_id=selected.id,
                            **({'history_omitted': omitted[0]} if omitted else {}),
                            **({'attachments': attachments} if attachments else {}))
                options = {'attachment_context': context_size} if context_size else {}
                async for token in model.stream(context, selected.id, **options):
                    answer += token
                    store.update_message(aid, answer, 'streaming')
                    yield event('delta', content=token)
                if not answer.strip():
                    raise RuntimeErrorLocal('The model finished without returning an answer. You can retry this response.')
                state = 'complete'
                store.update_message(aid, answer, state)
                yield event('done')
            except ContextLimitError as exc:
                state = 'error'
                if mid and not answer.strip():
                    # A rejected prompt belongs in the editable draft, not in history.
                    with store.connect() as db:
                        db.execute('DELETE FROM messages WHERE id IN (?, ?)', (mid, aid))
                        db.execute('UPDATE conversations SET title=?, updated_at=? WHERE id=? AND updated_at=?',
                                   (conversation['title'], conversation['updated_at'], cid, stamp))
                    for file in new_files:
                        remove_upload(uploads_dir, file.storage_name)
                    yield event('error', detail=str(exc), restore_draft=True)
                else:
                    yield event('error', detail=str(exc))
            except RuntimeErrorLocal as exc:
                state = 'error'
                store.update_message(aid, answer, state)
                yield event('error', detail=str(exc))
            except Exception:
                state = 'error'
                store.update_message(aid, answer, state)
                yield event('error', detail='Generation failed. You can retry this response.')
            finally:
                try:
                    store.update_message(aid, answer, state)
                finally:
                    generation_lock.release()
        return StreamingResponse(events(), media_type='application/x-ndjson')

    @app.post('/api/conversations/{cid}/messages')
    async def send(cid: str, body: Prompt):
        return await generate(cid, body.content, body.model_id)

    @app.post('/api/conversations/{cid}/messages/attachments')
    async def send_attachments(cid: str, request: Request):
        require_idle()
        require_conversation(cid)
        if request.headers.get('content-type', '').split(';', 1)[0].lower() != 'multipart/form-data':
            raise HTTPException(415, 'Use multipart/form-data for attachments.')
        try:
            form = await parse_upload(request)
            try:
                content, model_id = form.get('content', ''), form.get('model_id')
                if not isinstance(content, str) or (model_id is not None and not isinstance(model_id, str)):
                    raise HTTPException(422, 'Invalid message fields.')
                check_prompt_size(content)
                uploads = form.getlist('files')
                if not uploads:
                    raise HTTPException(422, 'Choose at least one attachment.')
                return await generate(cid, content, model_id, uploads)
            finally:
                await form.close()
        except AttachmentError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    @app.post('/api/conversations/{cid}/retry')
    async def retry(cid: str, body: ModelChoice | None = None):
        try:
            return await generate(cid, model_id=body.model_id if body else None)
        except AttachmentError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    app.include_router(dictation_router())

    if (ROOT / 'frontend' / 'dist').exists():
        app.mount('/', StaticFiles(directory=ROOT / 'frontend' / 'dist', html=True), name='frontend')
    return app


app = create_app()
