"""
storage.py
------------
WHY: Without persistence, every analysis disappears the moment the browser
tab closes — a user can't track whether their resume score is improving
over time. This adds a lightweight local SQLite history keyed by a random
per-browser session id (stored in Streamlit's session_state), so repeat
visits in the same session (and, on a persistent-disk deployment, across
sessions if you extend this to real login) show a trend.

NOTE (honest limitation): on ephemeral hosting (e.g. some free tiers of
Streamlit Community Cloud), the SQLite file can reset on redeploy/restart.
For a production multi-user product, swap this for a hosted DB (Postgres/
Supabase/etc.) — the function signatures here are written so that swap
only touches this one file.
"""
import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models", "history.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            target_role TEXT,
            predicted_role TEXT,
            ats_score REAL,
            num_skills INTEGER,
            num_missing INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(session_id: str, analysis: dict):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analysis_history (session_id, timestamp, target_role, predicted_role, ats_score, "
        "num_skills, num_missing) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, datetime.now().isoformat(timespec="seconds"), analysis["target_role"],
         analysis["pred_role"], analysis["ats_score"], len(analysis["found_skills"]), len(analysis["missing"]))
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 20):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM analysis_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
