# ingest_kb.py
import os, glob, re, hashlib, yaml, sys
from dataclasses import dataclass
from typing import List, Dict, Any

# ---- SQLite shim (needed on Streamlit Cloud) ----
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from markdown import markdown

# ------------- Config -------------
KB_DIR        = "kb"
DB_DIR        = "vectordb"
COLLECTION    = "finance_kb"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE        = "cpu"       # force CPU everywhere
RESET_COLLECTION = False    # set True to drop & recreate the KB once

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80          # keep modest; high overlap creates dupes
BATCH         = 128

# ------------- Helpers -------------
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_front_matter(text: str) -> (Dict[str, Any], str):
    """Extract YAML front matter at the very top (--- ... ---)."""
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
    # strip tags and collapse spaces
    txt = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", txt).strip()

def normalize_text(t: str) -> str:
    """Whitespace + punctuation light-normalization for stable hashing."""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    chunks = []
    step = max(1, size - max(0, overlap))
    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def content_id(source: str, chunk_text: str) -> str:
    """Stable ID from normalized content (prevents dupes across runs/files)."""
    key = f"{source}::{normalize_text(chunk_text)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def sanitize_meta(d: dict) -> dict:
    """Chroma metadata must be scalars/strings."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            out[k] = str(v)
        else:
            out[k] = v
    return out

@dataclass
class KBChunk:
    id: str
    text: str
    source: str
    meta: Dict[str, Any]

# ------------- Main ingest -------------
def ingest():
    # 0) Embedder (single instance; CPU)
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    # 1) Chroma client/collection
    client = chromadb.PersistentClient(path=DB_DIR)

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION)
            print(f"⚠️  Deleted existing collection '{COLLECTION}'")
        except Exception:
            pass

    coll = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # 2) Scan files → chunks → dedupe by content hash
    docs: List[KBChunk] = []
    seen_ids = set()

    files = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    if not files:
        print("No KB markdown files found in ./kb")
        return

    for path in files:
        raw = read_file(path)
        meta, body_md = parse_front_matter(raw)
        body_txt = md_to_text(body_md)
        pieces = chunk_text(body_txt)

        base = os.path.basename(path)
        for idx, piece in enumerate(pieces):
            cid = content_id(base, piece)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            docs.append(
                KBChunk(
                    id=cid,
                    text=piece,
                    source=base,
                    meta={"chunk": idx, **meta},
                )
            )

    if not docs:
        print("Nothing to ingest after dedupe.")
        return

    # 3) Embed + upsert (batched)
    texts      = [d.text for d in docs]
    ids        = [d.id   for d in docs]
    metadatas  = [sanitize_meta({"source": d.source, **d.meta}) for d in docs]

    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

    for i in range(0, len(ids), BATCH):
        coll.upsert(
            ids=ids[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
        )

    print(f"✅ Ingested {len(ids)} unique chunks from {len(set(d.source for d in docs))} files into {DB_DIR}/{COLLECTION}")

# ------------- Quick test query -------------
def preview(query: str, k: int = 5):
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_collection(COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=qvec, n_results=k, include=["documents","metadatas"])
    print("\nTop matches:")
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        doc  = res["documents"][0][i]
        title = meta.get("title") or "(kb)"
        print(f"- {title} [{meta.get('source')}#{meta.get('chunk')}]")
        print(f"  {doc[:160]}...")
    print()

if __name__ == "__main__":
    ingest()
    # smoke test
    preview("how much should my emergency fund be?")
