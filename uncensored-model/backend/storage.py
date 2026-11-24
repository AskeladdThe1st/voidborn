"""SQLite persistence with short-lived connections."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .attachments import StoredFile


class Store:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'complete', created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id);
                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL, mime TEXT NOT NULL, size INTEGER NOT NULL,
                    storage_name TEXT NOT NULL UNIQUE, status TEXT NOT NULL,
                    warning TEXT, passages TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS attachments_conversation ON attachments(conversation_id);
            ''')
            if db.execute('PRAGMA user_version').fetchone()[0] < 2:
                db.execute('PRAGMA user_version=2')
            db.execute("UPDATE messages SET status='interrupted' WHERE status='streaming'")

    def messages(self, cid):
        with self.connect() as db:
            messages = [dict(row) for row in db.execute(
                'SELECT * FROM messages WHERE conversation_id=? ORDER BY rowid', (cid,))]
            by_id = {message['id']: message for message in messages}
            for row in db.execute('''
                SELECT id, message_id, filename, mime, size, status, warning
                FROM attachments WHERE conversation_id=? ORDER BY rowid
            ''', (cid,)):
                entry = dict(row)
                by_id[entry.pop('message_id')].setdefault('attachments', []).append(entry)
            return messages

    def save_attachments(self, db, cid, mid, files):
        db.executemany('''
            INSERT INTO attachments
                (id, message_id, conversation_id, filename, mime, size, storage_name, status, warning, passages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(f.id, mid, cid, f.filename, f.mime, f.size, f.storage_name, f.status,
               f.warning, json.dumps(f.passages, ensure_ascii=False)) for f in files])

    def attachment_sources(self, cid, mid):
        with self.connect() as db:
            return [StoredFile(**{**dict(row), 'passages': json.loads(row['passages'])}) for row in db.execute('''
                SELECT id, filename, mime, size, storage_name, status, warning, passages
                FROM attachments WHERE conversation_id=? AND message_id=? ORDER BY rowid
            ''', (cid, mid))]

    def update_message(self, mid, content, status):
        with self.connect() as db:
            db.execute('UPDATE messages SET content=?, status=? WHERE id=?', (content, status, mid))
