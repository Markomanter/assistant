# db.py

import os
import sqlite3
import threading
from datetime import datetime

import config

# шлях до файлу БД (можеш змінити в config.py, якщо хочеш інший)
DB_PATH = getattr(config, "DB_PATH", "assistant.sqlite3")

_lock = threading.Lock()


def init_db() -> None:
    """Створює таблицю, якщо її ще немає."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    with _lock, sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_lang TEXT,
                user_text TEXT NOT NULL,
                assistant_think TEXT,
                assistant_reply TEXT NOT NULL
            );
            """
        )
        conn.commit()


def save_turn(
    user_text: str,
    user_lang: str | None,
    assistant_think: str | None,
    assistant_reply: str,
) -> None:
    """Записує один крок діалогу в БД."""
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    lang = (user_lang or "").lower()

    with _lock, sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO conversations
                (ts, user_lang, user_text, assistant_think, assistant_reply)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, lang, user_text, assistant_think or "", assistant_reply),
        )
        conn.commit()


# ініціалізуємо БД при імпорті модуля
init_db()
