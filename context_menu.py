from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable


PROJECT_DIR = Path(__file__).resolve().parent
QUEUE_DATABASE = Path(tempfile.gettempdir()) / "face-folder-sorter-context-menu.sqlite3"
QUIET_SECONDS = 0.8


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=10, isolation_level=None)
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS current_batch (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            token TEXT NOT NULL,
            updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS selected_files (
            token TEXT NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (token, path)
        );
        """
    )
    return connection


def launch_sorter(paths: list[str]) -> None:
    python = Path(sys.executable).with_name("python.exe")
    subprocess.Popen(
        [str(python), str(PROJECT_DIR / "main.py"), "--files", *paths],
        cwd=PROJECT_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        close_fds=True,
    )


def enqueue_file(
    path: Path,
    database: Path = QUEUE_DATABASE,
    quiet_seconds: float = QUIET_SECONDS,
    launcher: Callable[[list[str]], None] = launch_sorter,
) -> None:
    path = path.expanduser().resolve()
    now = time.time()
    connection = _connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT token, updated FROM current_batch WHERE id = 1"
        ).fetchone()
        stale_after = max(10.0, quiet_seconds * 4)
        is_leader = row is None or now - float(row[1]) > stale_after
        if is_leader:
            token = uuid.uuid4().hex
            connection.execute("DELETE FROM selected_files")
            connection.execute(
                """
                INSERT INTO current_batch(id, token, updated) VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET token = excluded.token,
                    updated = excluded.updated
                """,
                (token, now),
            )
        else:
            token = str(row[0])
            connection.execute(
                "UPDATE current_batch SET updated = ? WHERE id = 1", (now,)
            )
        connection.execute(
            "INSERT OR IGNORE INTO selected_files(token, path) VALUES (?, ?)",
            (token, str(path)),
        )
        connection.commit()
    finally:
        connection.close()

    if not is_leader:
        return

    while True:
        time.sleep(max(0.05, quiet_seconds / 2))
        connection = _connect(database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT token, updated FROM current_batch WHERE id = 1"
            ).fetchone()
            if row is None or row[0] != token:
                connection.rollback()
                return
            if time.time() - float(row[1]) < quiet_seconds:
                connection.rollback()
                continue
            paths = [
                item[0]
                for item in connection.execute(
                    "SELECT path FROM selected_files WHERE token = ? ORDER BY path",
                    (token,),
                )
            ]
            connection.execute(
                "DELETE FROM selected_files WHERE token = ?", (token,)
            )
            connection.execute(
                "DELETE FROM current_batch WHERE id = 1 AND token = ?", (token,)
            )
            connection.commit()
        finally:
            connection.close()
        if paths:
            launcher(paths)
        return


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    enqueue_file(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
