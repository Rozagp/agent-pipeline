# core/memory_store.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal
import json
import sqlite3
import threading
import time


MemoryBackend = Literal["json", "sqlite"]


@dataclass
class MemoryStore:
    def put_document(self, doc: Dict[str, Any]) -> str: ...
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]: ...
    def put_analysis(self, doc_id: str, task: str, result: Dict[str, Any]) -> int: ...
    def get_analysis(self, doc_id: str, task: Optional[str] = None) -> list[Dict[str, Any]]: ...


#JSON logic
@dataclass
class JsonStore(MemoryStore):
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "docs").mkdir(exist_ok=True)
        (self.root / "analyses").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def put_document(self, doc: Dict[str, Any]) -> str:
        doc_id = doc["id"]
        path = self.root / "docs" / f"{doc_id}.json"
        with self._lock:
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        path = self.root / "docs" / f"{doc_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put_analysis(self, doc_id: str, task: str, result: Dict[str, Any]) -> int:
        ts = int(time.time() * 1000)
        rec = {"id": ts, "doc_id": doc_id, "task": task, "result": result, "created_at": ts}
        path = self.root / "analyses" / f"{doc_id}__{task}__{ts}.json"
        with self._lock:
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return ts

    def get_analysis(self, doc_id: str, task: Optional[str] = None) -> list[Dict[str, Any]]:
        items = []
        prefix = f"{doc_id}__{task}__" if task else f"{doc_id}__"
        for p in sorted((self.root / "analyses").glob(f"{prefix}*.json")):
            items.append(json.loads(p.read_text(encoding="utf-8")))
        return items

#sqlite logic

@dataclass
class SqliteStore(MemoryStore):
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as cx:
            cx.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    tokens INTEGER,
                    meta_json TEXT NOT NULL,
                    sections_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            cx.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(id)
                )
            """)
            cx.commit()

    def _conn(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.path.as_posix(), timeout=30, check_same_thread=False)
        return cx

    def put_document(self, doc: Dict[str, Any]) -> str:
        with self._lock, self._conn() as cx:
            cx.execute(
                "REPLACE INTO documents (id, path, tokens, meta_json, sections_json, created_at) VALUES (?,?,?,?,?,?)",
                (
                    doc["id"],
                    doc.get("path"),
                    int(doc.get("tokens") or 0),
                    json.dumps(doc["meta"], ensure_ascii=False),
                    json.dumps(doc["sections"], ensure_ascii=False),
                    int(time.time() * 1000),
                ),
            )
            cx.commit()
        return doc["id"]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as cx:
            row = cx.execute(
                "SELECT id, path, tokens, meta_json, sections_json FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
        if not row:
            return None
        meta = json.loads(row[3])
        sections = json.loads(row[4])
        return {"id": row[0], "path": row[1], "tokens": row[2], "meta": meta, "sections": sections}

    def put_analysis(self, doc_id: str, task: str, result: Dict[str, Any]) -> int:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT INTO analyses (doc_id, task, result_json, created_at) VALUES (?,?,?,?)",
                (doc_id, task, json.dumps(result, ensure_ascii=False), int(time.time() * 1000)),
            )
            cx.commit()
            rid = cx.execute("SELECT last_insert_rowid()").fetchone()[0]
        return int(rid)

    def get_analysis(self, doc_id: str, task: Optional[str] = None) -> list[Dict[str, Any]]:
        q = "SELECT id, task, result_json, created_at FROM analyses WHERE doc_id=?"
        params: Tuple[Any, ...] = (doc_id,)
        if task:
            q += " AND task=?"
            params += (task,)
        q += " ORDER BY created_at DESC"
        out = []
        with self._conn() as cx:
            for rid, tsk, rjson, ts in cx.execute(q, params).fetchall():
                out.append({"id": rid, "doc_id": doc_id, "task": tsk, "result": json.loads(rjson), "created_at": ts})
        return out


def make_store(backend: MemoryBackend = "sqlite", root: Optional[Path] = None) -> MemoryStore:
    if backend == "json":
        return JsonStore(root or Path("./data_store"))
    return SqliteStore((root or Path("./data_store")) / "memory.db")
