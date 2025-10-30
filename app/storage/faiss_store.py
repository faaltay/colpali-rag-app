import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path(__file__).resolve().parent.parent / "faiss_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
EMBED_DIM = MODEL.get_sentence_embedding_dimension()


def _meta_path(collection: str) -> Path:
    return DATA_DIR / f"{collection}.db"


def _index_path(collection: str) -> Path:
    return DATA_DIR / f"{collection}.index"


class LocalFaissStore:
    def __init__(self):
        self._open_indexes: Dict[str, faiss.Index] = {}

    def _ensure_db(self, collection: str):
        dbp = _meta_path(collection)
        create = not dbp.exists()
        conn = sqlite3.connect(str(dbp))
        if create:
            conn.execute(
                "CREATE TABLE docs (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, metadata TEXT)"
            )
            conn.commit()
        conn.close()

    def get_collections(self) -> List[str]:
        return [p.stem for p in DATA_DIR.glob("*.db")]

    def create_collection(self, collection: str, *, dim: int = EMBED_DIM):
        self._ensure_db(collection)
        idx_p = _index_path(collection)
        if not idx_p.exists():
            base = faiss.IndexFlatIP(dim)
            idm = faiss.IndexIDMap(base)
            faiss.write_index(idm, str(idx_p))

    def _load_index(self, collection: str) -> faiss.Index:
        if collection in self._open_indexes:
            return self._open_indexes[collection]
        idx_p = _index_path(collection)
        if not idx_p.exists():
            base = faiss.IndexFlatIP(EMBED_DIM)
            idm = faiss.IndexIDMap(base)
            faiss.write_index(idm, str(idx_p))
        idx = faiss.read_index(str(idx_p))
        self._open_indexes[collection] = idx
        return idx

    def _save_index(self, collection: str):
        if collection in self._open_indexes:
            faiss.write_index(self._open_indexes[collection], str(_index_path(collection)))

    def add_texts(self, collection: str, texts: List[str], metadatas: Optional[List[dict]] = None):
        self._ensure_db(collection)
        conn = sqlite3.connect(str(_meta_path(collection)))
        cur = conn.cursor()
        metas = metadatas or [{} for _ in texts]

        embeddings = MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings.astype('float32'))

        idx = self._load_index(collection)
        if not isinstance(idx, faiss.IndexIDMap):
            base = idx
            idx = faiss.IndexIDMap(base)
            self._open_indexes[collection] = idx

        inserted_ids = []
        for txt, meta in zip(texts, metas):
            cur.execute("INSERT INTO docs (text, metadata) VALUES (?, ?)", (txt, json.dumps(meta)))
            inserted_ids.append(cur.lastrowid)
        conn.commit()

        ids_np = np.array(inserted_ids, dtype='int64')
        idx.add_with_ids(embeddings.astype('float32'), ids_np)
        self._save_index(collection)
        conn.close()

    def search_texts(self, collection: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        idx_p = _index_path(collection)
        if not idx_p.exists():
            return []

        q_emb = MODEL.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb.astype('float32'))
        idx = self._load_index(collection)
        D, I = idx.search(q_emb.astype('float32'), top_k)
        ids = I[0].tolist()
        conn = sqlite3.connect(str(_meta_path(collection)))
        cur = conn.cursor()
        results = []
        for score, id_ in zip(D[0].tolist(), ids):
            if id_ == -1:
                continue
            cur.execute("SELECT text, metadata FROM docs WHERE id = ?", (int(id_),))
            row = cur.fetchone()
            if not row:
                continue
            text, meta_json = row
            meta = json.loads(meta_json) if meta_json else {}
            results.append({"id": int(id_), "score": float(score), "text": text, "metadata": meta})
        conn.close()
        return results