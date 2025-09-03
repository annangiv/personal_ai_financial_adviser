# ingest_kb.py
import os, glob, re, uuid, yaml
from dataclasses import dataclass
from typing import List, Dict, Any

import sys, pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from markdown import markdown

# -------- config --------
KB_DIR = "kb"
DB_DIR = "vectordb"
COLLECTION = "finance_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"  
CHUNK_SIZE = 600   
CHUNK_OVERLAP = 80

# -------- helpers --------
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_front_matter(text: str) -> (Dict[str, Any], str):
    """Extract YAML front-matter at the very top (--- ... ---)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return meta, body
    return {}, text

def md_to_text(md: str) -> str:
    """Convert markdown → plain text (quick + simple)."""
    html = markdown(md)
    return re.sub("<[^<]+?>", "", html)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += max(1, size - overlap)
    return chunks

@dataclass
class KBChunk:
    id: str
    text: str
    source: str
    meta: Dict[str, Any]

# -------- main ingest --------
def ingest():
    # 1) load embedder
    embedder = SentenceTransformer(EMBED_MODEL)

    # 2) prepare chroma collection (persisted on disk)
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    docs: List[KBChunk] = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        raw = read_file(path)
        meta, body_md = parse_front_matter(raw)
        body_txt = md_to_text(body_md)
        pieces = chunk_text(body_txt)

        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            docs.append(
                KBChunk(
                    id=str(uuid.uuid4()),
                    text=piece,
                    source=os.path.basename(path),
                    meta={"chunk": idx, **meta},
                )
            )

    if not docs:
        print("No KB markdown files found in ./kb")
        return

    # 3) embed + upsert (batched)
    texts = [d.text for d in docs]
    ids = [d.id for d in docs]
    metadatas = [ _sanitize_meta({"source": d.source, **d.meta}) for d in docs ]
    embeddings = SentenceTransformer(EMBED_MODEL).encode(
        texts, normalize_embeddings=True
    ).tolist()

    BATCH = 128
    for i in range(0, len(ids), BATCH):
        coll.upsert(
            ids=ids[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
        )

    print(f"✅ Ingested {len(ids)} chunks from {len(set(d.source for d in docs))} files into {DB_DIR}/{COLLECTION}")

# -------- quick test query --------
def preview(query: str, k: int = 5):
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_collection(COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=qvec, n_results=k)
    print("\nTop matches:")
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        title = meta.get("title", "(no title)")
        print(f"- {title} [{meta.get('source')}#{meta.get('chunk')}]")
        print(f"  {doc[:160]}...")
    print()

def _sanitize_meta(d: dict) -> dict:
    """Chroma metadata must be scalar; coerce lists/dicts to strings."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = ", ".join(map(str, v))      
        elif isinstance(v, dict):
            out[k] = str(v)                        
        else:
            out[k] = v
    return out

if __name__ == "__main__":
    ingest()
    # quick smoke test
    preview("how much should my emergency fund be?")
