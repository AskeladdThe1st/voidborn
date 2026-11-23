"""SQLite persistence with short-lived connections."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path


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
                PRAGMA user_version=1;
            ''')
            db.execute("UPDATE messages SET status='interrupted' WHERE status='streaming'")

    def messages(self, cid):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT * FROM messages WHERE conversation_id=? ORDER BY rowid', (cid,))]

    def update_message(self, mid, content, status):
        with self.connect() as db:
            db.execute('UPDATE messages SET content=?, status=? WHERE id=?', (content, status, mid))
