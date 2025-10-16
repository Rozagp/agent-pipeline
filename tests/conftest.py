# tests/conftest.py
import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typing import Any, Dict, List
import pytest

class StoreDouble:
    def __init__(self):
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.an: Dict[str, List[Dict[str, Any]]] = {}

    def put_document(self, doc: Dict[str, Any]) -> str:
        did = doc.get("id") or f"doc{len(self.docs)+1}"
        self.docs[did] = doc
        return did

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        return self.docs.get(doc_id, {})

    def put_analysis(self, key: str, task: str, result: Dict[str, Any]) -> None:
        row = {"task": task, "data": result, "id": len(self.an.get(key, [])) + 1}
        self.an.setdefault(key, []).append(row)

    def get_analyses(self, doc_id: str) -> List[Dict[str, Any]]:
        return self.an.get(doc_id, [])

@pytest.fixture
def store():
    return StoreDouble()
