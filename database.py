"""
database.py - Defense-Grade SQLite Storage for Biometric Vectors and Metadata
Replaces vulnerable Python pickle serialization with structured, ACID-compliant,
zero-server SQLite embedded storage for watchlist vectors and audit trails.
"""

import os
import sqlite3
import numpy as np
import json


DB_PATH = "ibvap_surveillance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                notes TEXT,
                embedding BLOB NOT NULL,
                photo_count INTEGER DEFAULT 1,
                thumb_url TEXT,
                created_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                person TEXT NOT NULL,
                confidence_pct TEXT NOT NULL,
                confidence_raw REAL NOT NULL,
                threat_level TEXT NOT NULL,
                match_tier TEXT NOT NULL,
                source TEXT NOT NULL,
                snapshot_path TEXT
            );
        """)
    conn.close()


def save_person_sqlite(person_id, name, category, notes, embedding, photo_count=1, thumb_url="", created_at=""):
    init_db()
    conn = get_connection()
    emb_bytes = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
    with conn:
        conn.execute("""
            INSERT INTO watchlist (id, name, category, notes, embedding, photo_count, thumb_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                category=excluded.category,
                notes=excluded.notes,
                embedding=excluded.embedding,
                photo_count=excluded.photo_count,
                thumb_url=excluded.thumb_url,
                created_at=excluded.created_at;
        """, (person_id, name, category, notes, emb_bytes, photo_count, thumb_url, created_at))
    conn.close()


def delete_person_sqlite(person_id):
    init_db()
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM watchlist WHERE id = ?", (person_id,))
    conn.close()


def load_watchlist_sqlite():
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, category, notes, embedding, photo_count, thumb_url, created_at FROM watchlist")
    rows = cur.fetchall()
    conn.close()

    enrolled_faces = {}
    enrolled_meta = {}

    for row in rows:
        pid = row["id"]
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if emb.size % 512 == 0 and emb.size > 0:
            emb = emb.reshape(-1, 512)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norms, 1e-7)
        else:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = (emb / norm).reshape(1, 512)
            else:
                emb = np.zeros((1, 512), dtype=np.float32)
        enrolled_faces[pid] = emb

        enrolled_meta[pid] = {
            "id": pid,
            "name": row["name"],
            "category": row["category"],
            "notes": row["notes"] or "",
            "photo_count": row["photo_count"] or 1,
            "thumb_url": row["thumb_url"] or f"/static/thumbnails/{pid}.jpg",
            "created_at": row["created_at"]
        }

    return enrolled_faces, enrolled_meta


def migrate_existing_data():
    init_db()
    faces, meta = load_watchlist_sqlite()
    if len(faces) > 0:
        return

    meta_file = "watchlist_meta.json"
    db_file = "watchlist_db.pkl"

    if os.path.exists(meta_file) and os.path.exists(db_file):
        try:
            import pickle
            with open(meta_file, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            with open(db_file, "rb") as f:
                data = pickle.load(f)
                raw_faces = data.get("enrolled_faces", {})

            for pid, pmeta in raw_meta.items():
                if pid in raw_faces:
                    save_person_sqlite(
                        person_id=pid,
                        name=pmeta.get("name", "Unknown"),
                        category=pmeta.get("category", "Unknown"),
                        notes=pmeta.get("notes", ""),
                        embedding=raw_faces[pid],
                        photo_count=pmeta.get("photo_count", 1),
                        thumb_url=pmeta.get("thumb_url", ""),
                        created_at=pmeta.get("created_at", "")
                    )
            print("[INFO] Migrated legacy data to SQLite successfully.")
        except Exception as e:
            print(f"[WARN] SQLite migration warning: {e}")

if __name__ == "__main__":
    migrate_existing_data()
    f, m = load_watchlist_sqlite()
    print(f"SQLite Database active with {len(f)} enrolled identities.")
