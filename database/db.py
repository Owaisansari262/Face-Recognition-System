import os
import sqlite3
from contextlib import closing

import numpy as np


def get_connection(db_path):
    """Database file se connection kholta hai. Folder/file na ho to bana deta hai."""
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path):
    """persons table banata hai (agar pehle se nahi bani)."""
    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                face_encoding BLOB NOT NULL,
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_person(db_path, name, encoding):
    """
    Naam + encoding save karta hai.
    Agar yeh naam pehle se hai to sirf uski encoding update ho jati hai
    (dobara register karne par duplicate row nahi banti).
    """
    # numpy array -> bytes (128 float32 numbers = 512 bytes)
    encoding_bytes = np.asarray(encoding, dtype=np.float32).flatten().tobytes()

    init_db(db_path)  # table na ho to pehle bana lo

    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO persons (name, face_encoding) VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET face_encoding = excluded.face_encoding
            """,
            (name, encoding_bytes),
        )
        conn.commit()


def load_all_persons(db_path):
    """
    Database se sab registered log wapas laata hai.
    Return: dictionary {naam: encoding (numpy array)}
    Database khali ho to {} return hota hai.
    """
    init_db(db_path)

    with closing(get_connection(db_path)) as conn:
        rows = conn.execute("SELECT name, face_encoding FROM persons ORDER BY id").fetchall()

    persons = {}
    for name, encoding_bytes in rows:
        # bytes -> numpy array (save karne ka ulta)
        persons[name] = np.frombuffer(encoding_bytes, dtype=np.float32)

    return persons