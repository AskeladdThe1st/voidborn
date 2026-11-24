import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .runtime import OllamaRuntime, RuntimeErrorLocal, context_messages, model_from_catalog
from .storage import Store
from .models import REGISTRY

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
    content: str = Field(min_length=1, max_length=2500)
    model_id: str | None = None

    @field_validator('content')
    @classmethod
    def clean(cls, value):
        value = value.strip()
        if not value or len(value.encode('utf-8')) > 2500:
            raise ValueError('Message must contain 1–2500 UTF-8 bytes')
        return value


class ModelChoice(BaseModel):
    model_id: str | None = None


def create_app(db_path=None, runtime=None):
    store = Store(Path(db_path) if db_path else ROOT / 'data' / 'chat.sqlite3')
    model = runtime or OllamaRuntime()
    generation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        store.initialize()
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
        response.headers['Content-Security-Policy'] = "default-src 'self'; connect-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
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
        return await model.catalog()

    @app.post('/api/compare')
    async def compare(body: Prompt):
        require_idle()
        await generation_lock.acquire()
        try:
            catalog = await model.catalog()
            installed = [m for m in catalog['models'] if m['installed']]
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
            db.execute('DELETE FROM conversations WHERE id=?', (cid,))

    async def generate(cid, content=None, model_id=None):
        selected = await choose(model_id)
        require_idle()
        conversation = require_conversation(cid)
        history = store.messages(cid)
        if content is None:
            if not history or history[-1]['role'] != 'assistant' or history[-1]['status'] == 'complete':
                raise HTTPException(409, 'There is no interrupted response to retry.')
        elif history and history[-1]['status'] != 'complete':
            raise HTTPException(409, 'Retry the interrupted response before sending another message.')
        await generation_lock.acquire()
        aid, stamp = str(uuid4()), now()
        try:
            with store.connect() as db:
                if content is None:
                    db.execute('DELETE FROM messages WHERE id=?', (history[-1]['id'],))
                else:
                    db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)', (str(uuid4()), cid, 'user', content, 'complete', stamp))
                    if not history and conversation['title'] == 'New conversation':
                        db.execute('UPDATE conversations SET title=? WHERE id=?', (content[:50], cid))
                db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)', (aid, cid, 'assistant', '', 'streaming', stamp))
                db.execute('UPDATE conversations SET updated_at=? WHERE id=?', (stamp, cid))
            context = context_messages(store.messages(cid))
        except BaseException:
            generation_lock.release()
            raise

        async def events():
            answer, state = '', 'interrupted'
            def event(kind, **data):
                return json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n'
            try:
                yield event('start', message_id=aid, model_id=selected.id)
                async for token in model.stream(context, selected.id):
                    answer += token
                    store.update_message(aid, answer, 'streaming')
                    yield event('delta', content=token)
                state = 'complete'
                store.update_message(aid, answer, state)
                yield event('done')
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

    @app.post('/api/conversations/{cid}/retry')
    async def retry(cid: str, body: ModelChoice | None = None):
        return await generate(cid, model_id=body.model_id if body else None)

    if (ROOT / 'frontend' / 'dist').exists():
        app.mount('/', StaticFiles(directory=ROOT / 'frontend' / 'dist', html=True), name='frontend')
    return app


app = create_app()
